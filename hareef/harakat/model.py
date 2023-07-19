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
from diacritization_evaluation.util import extract_haraqat
from hareef.learning_rates import adjust_learning_rate
from hareef.utils import (
    calculate_error_rates,
    format_error_rates_as_table,
    categorical_accuracy,
)
from lightning.pytorch import LightningModule

from .diacritizer import TorchDiacritizer
from .dataset import load_validation_data, load_test_data
from .modules.attentions import MultiHeadAttention, sequence_mask
from .modules.positional_encoding import PositionalEncoding


_LOGGER = logging.getLogger(__package__)



class CustomGRU(nn.Module):
    def __init__(self, *args, vertical_dropout=0.0, use_layernorm=False, pad_idx=0.0, **kwargs):
        super().__init__()
        self.dropout = output_dropout = kwargs.pop("dropout", 0.0)
        self.gru = nn.GRU(*args, dropout=vertical_dropout, **kwargs)
        self.pad_idx = float(pad_idx)
        self.batch_first = self.gru.batch_first
        self.dropout = nn.Dropout(output_dropout)
        if use_layernorm:
            bidirectional = self.gru.bidirectional
            hidden_size = self.gru.hidden_size
            normalized_shape = (hidden_size * 2) if bidirectional else hidden_size
            self.layernorm = nn.LayerNorm(
                normalized_shape=normalized_shape,
                eps=1e-6
            )
        else:
            self.layernorm = nn.Identity()

    def forward(self, input: torch.Tensor, lengths: torch.Tensor, hx: Optional[torch.Tensor]=None) -> torch.Tensor:
        packed_input = pack_padded_sequence(input, lengths, batch_first=self.batch_first)
        output, hx = self.gru(packed_input, hx)
        output, _lengths = pad_packed_sequence(output, batch_first=self.batch_first, padding_value=self.pad_idx)
        output = self.layernorm(output)
        return self.dropout(output.tanh())


class TwosDiacEmbedding(nn.Module):

    def __init__(self, vocab_size, dim, max_len, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.pos_enc = PositionalEncoding(dim, dropout_p=0.1, max_len=max_len)

    def forward(self, src):
        embed_out = self.embedding(src)
        lin_out = self.lin(embed_out)
        lin_weighted = lin_out * 16.000000
        return self.pos_enc(lin_weighted.permute(0, 2, 1)).permute(0, 2, 1)


class TwosDiacEncoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attn_layernorm = nn.LayerNorm(normalized_shape=input_dim)
        self.attn =             MultiHeadAttention(
            channels=input_dim,
            out_channels=input_dim,
            n_heads=4,
            p_dropout=0.1,
            proximal_bias=True,
            proximal_init=True
        )
        self.attn_lin = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_dim),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x, z, lengths):
        layernorm_out = self.attn_layernorm(x)

        input_mask = torch.unsqueeze(
            sequence_mask(lengths, layernorm_out.size(1)), 1
        ).type_as(layernorm_out)

        attn_input = layernorm_out.permute(0, 2, 1) * input_mask
        attn_mask = input_mask.unsqueeze(2) * input_mask.unsqueeze(-1)
        attn_out = self.attn(attn_input, attn_input, attn_mask=attn_mask)
        attn_out = attn_out.permute(0, 2, 1)

        attn_out = attn_out + z

        output = self.ff(attn_out) + attn_out

        return output, attn_out


class TwosDiacDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, max_len):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=input_dim)
        self.gru = CustomGRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.lin = nn.Linear(input_dim * 2, input_dim, bias=False)
        # self.pos_enc = PositionalEncoding(input_dim, max_len=max_len)
        self.dropout = nn.Dropout(0.1)
        self.projections = nn.Linear(input_dim, output_dim)

    def forward(self, x, lengths):
        layernorm_out = self.layernorm(x)
        gru_out = self.gru(layernorm_out, lengths)
        lin_out = self.lin(gru_out)
        lin_weighted = lin_out * 16.000000
        #output = self.pos_enc(lin_out.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout(lin_out)
        return self.projections(output).log_softmax(dim=2)


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
            embedding_dim=config["embedding_dim"],
            max_len=config["max_len"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        max_len,
        input_pad_idx,
        target_pad_idx,
    ):
        self.source_embed = TwosDiacEmbedding(
            vocab_size=inp_vocab_size,
            dim=256,
            max_len=max_len,
            padding_idx=input_pad_idx
        )
        self.source_embed_diac = TwosDiacEmbedding(
            vocab_size=targ_vocab_size,
            dim=256,
            max_len=max_len,
            padding_idx=target_pad_idx
        )
        self.encoder_layers = nn.ModuleList([
            TwosDiacEncoder(256, 256)
            for i in range(6)
        ])
        self.decoder = TwosDiacDecoder(256, targ_vocab_size, max_len=max_len)

    def forward(self, char_inputs: torch.Tensor, diac_inputs: torch.Tensor, input_lengths: torch.Tensor):
        embed_out = self.source_embed(char_inputs) + self.source_embed_diac(diac_inputs)

        enc_out = prev_attn = embed_out
        for enc in self.encoder_layers:
            enc_out, prev_attn = enc(enc_out, prev_attn, input_lengths)

        predictions = self.decoder(enc_out, input_lengths)

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
            min_lr=self.config["min_lr"]
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
                num_batches = self.config["error_rates_n_batches"],
                predictions_dir=Path(self.trainer.log_dir).joinpath("predictions")
            )
            _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))
            if self.logger is not None:
                num_batches = max(
                    self.config["n_predicted_text_tensorboard"] // self.config["batch_size"],
                    1
                )
                for i, (gt, pred) in enumerate(self.example_predictions(data_loader, diacritizer, num_batches)):
                    self.logger.experiment.add_text(f"example-text/{i}", f"{gt} |->  {pred}")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        diacritizer = TorchDiacritizer(self.config, take_hints=False, model=self)
        error_rates = self.evaluate_with_error_rates(
            diacritizer,
            data_loader=data_loader,
            num_batches = self.config["error_rates_n_batches"],
            predictions_dir=Path(self.trainer.log_dir).joinpath("predictions")
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
        batch["lengths"] = batch["lengths"].to('cpu')
        outputs = self(batch["src"], batch["diac"], batch["lengths"])
        predictions = outputs["diacritics"].contiguous()
        targets = batch["target"].contiguous()
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        diac_loss = self.diacritics_loss(predictions.to(self.device), targets.to(self.device))
        diac_accuracy = categorical_accuracy(predictions.to(self.device), targets.to(self.device), self.target_pad_idx, device=self.device)
        return {
            "loss": diac_loss,
            "accuracy": diac_accuracy,
        }

    @classmethod
    def evaluate_with_error_rates(cls, diacritizer, data_loader, num_batches, predictions_dir, hint_p=None):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        all_orig = []
        all_predicted = []
        results = {}
        for batch in more_itertools.take(num_batches, data_loader):
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
            for i in random.sample(range(diac_len),  k=round(diac_len * mask_p)):
                diac[i] = ""
            results.append(
                "".join(more_itertools.interleave(chars, diac))
            )

        return results
