# coding: utf-8

import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from hareef.learning_rates import adjust_learning_rate
from hareef.utils import (
    calculate_error_rates,
    format_error_rates_as_table,
    categorical_accuracy,
)
from lightning.pytorch import LightningModule
from torch import nn, optim

from .diacritizer import TorchDiacritizer
from .dataset import load_validation_data, load_test_data
from .modules.k_lstm import K_LSTM
from .modules.attentions import MultiHeadAttention



_LOGGER = logging.getLogger(__package__)


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
            inp_vocab_size=config.len_input_symbols,
            targ_vocab_size=config.len_target_symbols,
            embedding_dim=config["embedding_dim"],
            max_len=config["max_len"],
            lstm_info=config["lstm_info"],
            lstm_dropout=config["lstm_dropout"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        max_len,
        lstm_info,
        lstm_dropout,
        input_pad_idx,
        target_pad_idx,
    ):
        (
            (lstm1_dim, lstm1_num_layers),
            (lstm2_dim, lstm2_num_layers),
            (lstm3_dim, lstm3_num_layers),
        ) = lstm_info
        self.char_embedding = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=input_pad_idx
        )
        self.lstm1 = K_LSTM(
            input_size=embedding_dim,
            hidden_size=lstm1_dim,
            num_layers=lstm1_num_layers,
            batch_first=True,
            bidirectional=True,
            recurrent_activation="hard_sigmoid",
            recurrent_dropout=lstm_dropout,
            vertical_dropout=lstm_dropout,
            return_states=True
        )
        self.layernorm1 = nn.LayerNorm(
            normalized_shape=lstm1_dim * 2,
            eps=1e-6
        )
        # self.batchnorm1 = nn.BatchNorm1d(
        # num_features=lstm1_dim * 2,
        # eps=1e-3,
        # momentum=0.99
        # )
        self.lstm2 = K_LSTM(
            input_size=lstm1_dim * 2,
            hidden_size=lstm2_dim,
            num_layers=lstm2_num_layers,
            batch_first=True,
            bidirectional=True,
            recurrent_activation="hard_sigmoid",
            recurrent_dropout=lstm_dropout,
            vertical_dropout=lstm_dropout,
            return_states=True
        )
        self.layernorm2 = nn.LayerNorm(
            normalized_shape=lstm2_dim * 2,
            eps=1e-6
        )
        self.lstm3 = K_LSTM(
            input_size=lstm2_dim * 2,
            hidden_size=lstm3_dim,
            num_layers=lstm3_num_layers,
            batch_first=True,
            bidirectional=True,
            recurrent_activation="hard_sigmoid",
            recurrent_dropout=lstm_dropout,
            vertical_dropout=lstm_dropout,
            return_states=True
        )
        self.attention = MultiHeadAttention(
            channels=lstm3_dim * 2,
            out_channels=lstm3_dim * 2,
            n_heads=6,
            p_dropout=lstm_dropout,
            proximal_init=True
        )
        self.projections = nn.Linear(lstm3_dim * 2, targ_vocab_size)

    def forward(self, src):
        char_embedding_out = self.char_embedding(src)
        lstm1_out, lstm1_state = self.lstm1(char_embedding_out)
        lstm1_out = self.layernorm1(lstm1_out.tanh())
        lstm2_out, lstm2_state =  self.lstm2(lstm1_out)
        lstm2_out = self.layernorm2(lstm2_out.tanh())
        lstm3_out, lstm3_state = self.lstm3(lstm2_out)
        lstm3_out = lstm3_out.tanh()
        attn_input = lstm3_out.permute(0, 2, 1)
        attn_out = self.attention(attn_input, attn_input)
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
            eps=1e-7
        )
        lr_factor = 0.2
        lr_patience = 3
        min_lr = 1e-7
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if (self.current_epoch + 1) % self.config[
            "evaluate_with_error_rates_epoches"
        ] == 0:
            data_loader = load_validation_data(self.config)
            error_rates = self.evaluate_with_error_rates(
                data_loader, Path(self.trainer.log_dir).joinpath("predictions")
            )
            _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        error_rates = self.evaluate_with_error_rates(
            data_loader, Path(self.trainer.log_dir).joinpath("predictions")
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
        outputs = self(batch["src"])
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

    def evaluate_with_error_rates(self, data_loader, predictions_dir):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        diacritizer = TorchDiacritizer(self.config)
        diacritizer.set_model(self)
        all_orig = []
        all_predicted = []
        results = {}
        num_processed = 0
        for batch in data_loader:
            for text in batch["original"]:
                if num_processed > self.config["error_rates_n_batches"]:
                    break
                all_orig.append(text)
                predicted, __ = diacritizer.diacritize_text(text)
                all_predicted += predicted
                num_processed += 1
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
        if self.logger is not None:
            num_examples = self.config["n_predicted_text_tensorboard"]
            for i, (org, pred) in enumerate(
                more_itertools.take(num_examples, zip(all_orig, all_predicted))
            ):
                self.logger.experiment.add_text(f"eval-text/{i}", f"{org} |->  {pred}")
        return results


"""
    def _build_layers(self, inp_vocab_size, targ_vocab_size, embedding_dim, lstm_dims, lstm_dropout):
        (lstm1_h, lstm1_n_layers), (lstm2_h, lstm2_n_layers), (lstm3_h, lstm3_n_layers) = lstm_dims
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim, padding_idx=self.pad_idx)
        self.bidirectional_1 = LSTMHardSigmoid(
            input_size=embedding_dim,
            hidden_size=lstm1_h,
            num_layers=lstm1_n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        self.batch_normalization_1 = nn.BatchNorm1d(
            num_features=lstm1_h * 2,
            eps = 1e-3,
            momentum=0.1,
        )
        self.bidirectional_2 = LSTMHardSigmoid(
            input_size=self.batch_normalization_1.num_features,
            hidden_size=lstm2_h,
            num_layers=lstm2_n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        self.bidirectional_3 = LSTMHardSigmoid(
            input_size=lstm2_h * 2,
            hidden_size=lstm3_h,
            num_layers=lstm3_n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        self.projections = nn.Linear(lstm3_h * 2, targ_vocab_size)

    def forward(self, src):
        embedding_out = self.embedding(src)
        lstm1_out, lstm1_h = self.bidirectional_1(embedding_out)
        lstm1_out = lstm1_out.tanh()
        batchnorm_out = self.batch_normalization_1(lstm1_out.transpose(1, 2))
        lstm2_out, lstm2_h = self.bidirectional_2(batchnorm_out.transpose(1, 2))
        lstm2_out = lstm2_out.tanh()
        lstm3_out, lstm2_h = self.bidirectional_3(lstm2_out)
        lstm3_out = lstm3_out.tanh()
        predictions = self.projections(lstm3_out).log_softmax(dim=1)
        return {
            "diacritics": predictions
        }
"""
