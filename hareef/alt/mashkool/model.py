# coding: utf-8

import logging
import os
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


_LOGGER = logging.getLogger(__package__)


class MashkoolGRU(nn.Module):
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


class MashkoolModel(LightningModule):
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
            gru_info=config["gru_info"],
            gru_dropout=config["gru_dropout"],
        attention_num_heads=config["attention_num_heads"],
            attention_dropout=config["attention_dropout"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        max_len,
        gru_info,
        gru_dropout,
        attention_num_heads,
        attention_dropout,
        input_pad_idx,
        target_pad_idx,
    ):
        (
            (gru1_dim, gru1_num_layers),
            (gru2_dim, gru2_num_layers),
            (gru3_dim, gru3_num_layers),
        ) = gru_info
        self.char_embedding = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=input_pad_idx
        )
        self.gru_layers = nn.ModuleList([
            MashkoolGRU(
                input_size=embedding_dim,
                hidden_size=gru1_dim,
                num_layers=gru1_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=gru_dropout,
                use_layernorm=True,
                pad_idx=input_pad_idx,
            ),
            MashkoolGRU(
                input_size=gru1_dim * 2,
                hidden_size=gru2_dim,
                num_layers=gru2_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=gru_dropout,
                use_layernorm=True,
                pad_idx=input_pad_idx,
            ),
            MashkoolGRU(
                input_size=gru2_dim * 2,
                hidden_size=gru3_dim,
                num_layers=gru3_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=gru_dropout,
                use_layernorm=True,
                pad_idx=input_pad_idx,
            ),
        ])
        gru_output_dim = gru3_dim * 2
        self.attention = MultiHeadAttention(
            channels=gru_output_dim,
            out_channels=gru_output_dim,
            n_heads=attention_num_heads,
            p_dropout=attention_dropout,
            proximal_bias=True,
            proximal_init=True,
            window_size=max_len // attention_num_heads
        )
        self.projections = nn.Linear(gru_output_dim, targ_vocab_size)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor):
        char_embedding_out = self.char_embedding(src)

        gru_out = char_embedding_out
        for gru in self.gru_layers:
            gru_out = gru(gru_out, lengths)

        input_mask = torch.unsqueeze(
            sequence_mask(lengths, gru_out.size(1)), 1
        ).type_as(gru_out)
        attn_input = gru_out.permute(0, 2, 1) * input_mask
        attn_mask = input_mask.unsqueeze(2) * input_mask.unsqueeze(-1)

        attn_out = self.attention(attn_input, attn_input, attn_mask=attn_mask)
        attn_out = attn_out.permute(0, 2, 1)

        predictions = self.projections(attn_out).log_softmax(dim=2)
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
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            betas=tuple(self.config["adam_betas"]),
            eps=self.config["adamw_eps"]
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
            diacritizer = TorchDiacritizer(self.config, model=self)
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
        diacritizer = TorchDiacritizer(self.config, model=self)
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
        outputs = self(batch["src"], batch["lengths"])
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

    @staticmethod
    def evaluate_with_error_rates(diacritizer, data_loader, num_batches, predictions_dir):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        all_orig = []
        all_predicted = []
        results = {}
        for batch in more_itertools.take(num_batches, data_loader):
            all_orig.extend(batch["original"])
            predicted, __ = diacritizer.diacritize_text(batch["original"])
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
