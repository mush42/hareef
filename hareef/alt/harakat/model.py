# coding: utf-8

import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import more_itertools
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from diacritization_evaluation.util import extract_haraqat
from hareef.learning_rates import adjust_learning_rate
from hareef.utils import (
    sequence_mask,
    calculate_error_rates,
    format_error_rates_as_table,
    categorical_accuracy,
)
from lightning.pytorch import LightningModule

from .diacritizer import TorchDiacritizer
from .dataset import load_validation_data, load_test_data
from .modules.pos_encoding import PositionalEncoding

from x_transformers.x_transformers import Attention


_LOGGER = logging.getLogger(__package__)


class HGRU(nn.Module):
    def __init__(
        self,
        *args,
        num_layers,
        vertical_dropout=0.0,
        use_layernorm=False,
        pad_idx=0.0,
        sum_bidi_output: Optional[bool]=True,
        use_pos_encoding: Optional[bool]=False,
        **kwargs,
    ):
        super().__init__()
        self.dropout = output_dropout = kwargs.pop("dropout", 0.0)
        self.gru = nn.GRU(*args, dropout=vertical_dropout, **kwargs)
        self.pad_idx = float(pad_idx)
        self.sum_bidi_output = sum_bidi_output
        self.use_pos_encoding = use_pos_encoding
        self.batch_first = self.gru.batch_first
        self.dropout = nn.Dropout(output_dropout)
        self.hidden_size = self.gru.hidden_size
        self.bidirectional = self.gru.bidirectional
        output_dim = self.hidden_size if sum_bidi_output else self.hidden_size * 2
        if use_layernorm:
            self.layernorm = nn.LayerNorm(normalized_shape=output_dim)
        else:
            self.layernorm = nn.Identity()
        self.pos_encoder = RotaryPositionalEncoding(output_dim)

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        packed_input = pack_padded_sequence(
            input, lengths, batch_first=self.batch_first
        )
        output, hx = self.gru(packed_input, hx)
        output, _lengths = pad_packed_sequence(
            output, batch_first=self.batch_first, padding_value=self.pad_idx
        )
        if self.sum_bidi_output and self.bidirectional:
            output = output[:, :, : self.hidden_size] + output[:, :, self.hidden_size :]
        output = self.layernorm(output)
        if self.use_pos_encoding:
            mask = torch.unsqueeze(
                sequence_mask(lengths, output.size(1)).bool(),
                1
            )
            pos_enc_input = output
            rope_out, _ = self.pos_encoder(lengths.max(), output.device)
            rope_out = rope_out
            output = apply_rotary_pos_emb(output, rope_out) + output
        return self.dropout(output.tanh())


class HEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, max_len, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.pos_enc = PositionalEncoding(
            channels=dim,
            max_len=max_len,
        )

    def forward(self, src, lengths):
        embed_out = self.embedding(src).to(src.device)
        lin_out = self.lin(embed_out)
        lin_weighted = lin_out * 16.000000
        pose_enc_input = lin_weighted.permute(0, 2, 1)
        pos_enc_mask = torch.unsqueeze(
            sequence_mask(lengths, lin_weighted.shape[1]), 1
        ).bool().to(src.device)
        return self.pos_enc(pose_enc_input, mask=pos_enc_mask).permute(0, 2, 1) + lin_weighted


class HEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attn_layernorm = nn.LayerNorm(input_dim)
        self.attention = Attention(
            dim=input_dim,
            heads=6,
            dropout=0.1,
            flash=True,
            onnxable=True,
        )
        self.attn_lin = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, output_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, residual, lengths):
        layernorm_out = self.attn_layernorm(x).to(x.device)

        attn_mask = sequence_mask(lengths, layernorm_out.shape[-1]).to(x.device)
        attn_out , __ = self.attention(layernorm_out)

        attn_out = attn_out + residual

        output = self.ff(attn_out) + attn_out

        return output, attn_out


class HOutput(nn.Module):
    def __init__(self, input_dim, output_dim, max_len):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.pos_enc1 = PositionalEncoding(
            channels=input_dim,
            max_len=max_len
        )
        self.lin = nn.Linear(input_dim, input_dim, bias=False)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.pos_enc2 = PositionalEncoding(
            channels=input_dim,
            max_len=max_len
        )
        self.projections = nn.Linear(input_dim, output_dim)

    def forward(self, x, lengths):
        pos_enc_mask = torch.unsqueeze(
            sequence_mask(lengths, x.shape[1]), 1
        ).bool().to(x.device)

        layernorm1_out = self.layernorm1(x).to(x.device)
        pose_enc1_input = layernorm1_out.permute(0, 2, 1)
        pos_enc1_out = self.pos_enc1(pose_enc1_input, mask=pos_enc_mask)
        pos_enc1_out = pos_enc1_out.permute(0, 2, 1) + layernorm1_out

        lin_out = self.lin(pos_enc1_out)
        lin_weighted = lin_out * 16.000000
        layernorm2_out = self.layernorm2(lin_weighted).to(x.device)

        pos_enc2_input = layernorm2_out.permute(0, 2, 1)
        pos_enc2_out = self.pos_enc2(pos_enc2_input, mask=pos_enc_mask)
        pos_enc2_out = pos_enc2_out.permute(0, 2, 1) + layernorm2_out

        return self.projections(pos_enc2_out).log_softmax(dim=2)


class HarakatModel(LightningModule):
    """ """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_pad_idx = self.config.text_encoder.input_pad_id
        self.target_pad_idx = self.config.text_encoder.target_pad_id

        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.diacritics_loss = nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self._build_layers(
            inp_vocab_size=config.len_input_symbols * 3,
            targ_vocab_size=config.len_target_symbols,
            model_dim=config["model_dim"],
            max_len=config["max_len"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        model_dim,
        max_len,
        input_pad_idx,
        target_pad_idx,
    ):
        self.source_embed = HEmbedding(
            vocab_size=inp_vocab_size,
            dim=model_dim,
            max_len=max_len,
            padding_idx=input_pad_idx,
        )
        # self.source_embed_diac = TwosDiacEmbedding(
        #    vocab_size=targ_vocab_size,
        #    dim=model_dim,
        #    max_len=max_len,
        #    padding_idx=target_pad_idx
        # )
        enc_num_layers = 4
        self.encoder_layers = nn.ModuleList(
            [
                HEncoder(
                    model_dim,
                    model_dim,
                )
                for i in range(enc_num_layers)
            ]
        )
        self.output = HOutput(model_dim, targ_vocab_size, max_len=max_len)

    def forward(
        self,
        char_inputs: torch.Tensor,
        diac_inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        embed_out = self.source_embed(
            char_inputs.to(self.device), input_lengths
        )  # + self.source_embed_diac(diac_inputs, input_lengths)

        enc_out = residual_attn = embed_out
        for enc in self.encoder_layers:
            enc_out, residual_attn = enc(enc_out, residual_attn, input_lengths)

        predictions = self.output(enc_out, input_lengths)

        return {"diacritics": predictions}

    def training_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.training_step_outputs.setdefault(name, []).append(val)
            self.log(name, val)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.val_step_outputs.setdefault(f"val_{name}", []).append(val)
            self.log(f"val_{name}", val)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.test_step_outputs.setdefault(f"test_{name}", []).append(val)
            self.log(f"test_{name}", val)
        return metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            betas=tuple(self.config["adam_betas"]),
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.config["lr_factor"],
            patience=self.config["lr_patience"],
            min_lr=self.config["min_lr"],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if (self.current_epoch + 1) % self.config[
            "evaluate_with_error_rates_epochs"
        ] == 0:
            data_loader = load_validation_data(self.config)
            diacritizer = TorchDiacritizer(self.config, take_hints=False, model=self)
            error_rates = self.evaluate_with_error_rates(
                diacritizer,
                data_loader=data_loader,
                num_batches=self.config["error_rates_n_batches"],
                predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
            )
            _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))
            if self.logger is not None:
                num_batches = max(
                    self.config["n_predicted_text_tensorboard"]
                    // self.config["batch_size"],
                    1,
                )
                for i, (gt, pred) in enumerate(
                    self.example_predictions(data_loader, diacritizer, num_batches)
                ):
                    self.logger.experiment.add_text(
                        f"example-text/{i}", f"{gt} |->  {pred}"
                    )

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        diacritizer = TorchDiacritizer(self.config, take_hints=False, model=self)
        error_rates = self.evaluate_with_error_rates(
            diacritizer,
            data_loader=data_loader,
            num_batches=self.config["error_rates_n_batches"],
            predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
        )
        _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))

    def _log_epoch_metrics(self, metrics):
        for name, values in metrics.items():
            epoch_metric_mean = torch.stack(values).mean()
            self.log(name, epoch_metric_mean)
            values.clear()

    def _process_batch(self, batch):
        batch["src"] = batch["src"].to(self.device)
        batch["target"] = batch["target"].to(self.device)
        batch["lengths"] = batch["lengths"].to("cpu")
        outputs = self(batch["src"], batch["diac"], batch["lengths"])
        predictions = outputs["diacritics"].contiguous()
        targets = batch["target"].contiguous()

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)

        diac_loss = self.diacritics_loss(
            predictions.to(self.device), targets.to(self.device)
        )
        diac_accuracy = categorical_accuracy(
            predictions.to(self.device),
            targets.to(self.device),
            self.target_pad_idx,
            device=self.device,
        )

        return {
            "loss": diac_loss,
            "accuracy": diac_accuracy,
        }

    @classmethod
    def evaluate_with_error_rates(
        cls, diacritizer, data_loader, num_batches, predictions_dir, hint_p=None
    ):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        all_orig = []
        all_predicted = []
        results = {}
        for batch in tqdm(
            more_itertools.take(num_batches, data_loader),
            total=num_batches,
            desc="Predicting",
            unit="batch",
        ):
            gt_lines = batch["original"]
            if hint_p:
                gt_lines = cls.apply_hint_mask(gt_lines, hint_p)
            predicted, __ = diacritizer.diacritize_text(gt_lines)
            all_orig.extend(gt_lines)
            all_predicted.extend(predicted)

        orig_path = os.path.join(predictions_dir, f"original.txt")
        with open(orig_path, "w", encoding="utf8") as file:
            lines = "\n".join(sent for sent in all_orig)
            file.write(lines)

        predicted_path = os.path.join(predictions_dir, f"predicted.txt")
        with open(predicted_path, "w", encoding="utf8") as file:
            lines = "\n".join(sent for sent in all_predicted)
            file.write(lines)

        try:
            results = calculate_error_rates(orig_path, predicted_path)
        except:
            _LOGGER.error("Failed to calculate DER/WER statistics", exc_info=True)
            results = {"DER": 0.0, "WER": 0.0, "DER*": 0.0, "WER*": 0.0}

        return results

    def example_predictions(self, data_loader, diacritizer, num_batches):
        gt_lines = []
        pred_lines = []
        for batch in more_itertools.take(num_batches, data_loader):
            gt_lines.extend(batch["original"])
            predictions, _ = diacritizer.diacritize_text(batch["original"])
            pred_lines.extend(predictions)

        yield from zip(gt_lines, pred_lines)

    @classmethod
    def apply_hint_mask(cls, lines, hint_p):
        mask_p = 1 - hint_p

        results = []
        for line in lines:
            __, chars, diac = extract_haraqat(line)
            diac_len = len(diac)
            for i in random.sample(range(diac_len), k=round(diac_len * mask_p)):
                diac[i] = ""
            results.append("".join(more_itertools.interleave(chars, diac)))

        return results
