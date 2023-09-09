# coding: utf-8

import logging
import os
import random
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import more_itertools
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from diacritization_evaluation.util import extract_haraqat
from hareef.utils import (
    sequence_mask,
    calculate_error_rates,
    format_error_rates_as_table,
    categorical_accuracy,
)
from lightning.pytorch import LightningModule

from .diacritizer import TorchDiacritizer
from .dataset import load_validation_data, load_test_data
from .modules.x_transformers import TransformerWrapper, Encoder


_LOGGER = logging.getLogger(__package__)


class NabihModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_pad_idx = self.config.text_encoder.input_pad_id
        self.target_pad_idx = self.config.text_encoder.target_pad_id

        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self._build_layers(
            inp_vocab_size=self.config.len_input_symbols * 3,
            targ_vocab_size=self.config.len_target_symbols,
            embedding_dim=self.config["embedding_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            max_len=self.config["max_len"],
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        num_layers,
        num_heads,
        max_len,
    ):
        self.transformer = TransformerWrapper(
            num_tokens = inp_vocab_size,
            max_seq_len = max_len,
            post_emb_norm=True,
            attn_layers=Encoder(
                dim = embedding_dim,
                depth=num_layers,
                heads=num_heads,
                layer_dropout=0.15,
                attn_dropout=0.1,
                ff_dropout=0.1,
                attn_flash=True,
                cascading_heads=True,
                rotary_pos_em=True,
                use_simple_rmsnorm = True,
                attn_gate_values = True,
                onnxable=True,
            )
        )
        self.fc_out = nn.Linear(inp_vocab_size, targ_vocab_size)

    def forward(self, inputs, lengths):
        x = inputs.to(self.device)
        lengths = lengths.to(self.device)

        mask = sequence_mask(lengths, max_length=x.shape[-1]).to(self.device)
        x = self.transformer(x, mask=mask)
        x = self.fc_out(x)
        return x

    def predict(self, inputs, lengths):
        output = self(inputs, lengths)
        logits = output.softmax(dim=2)
        predictions = torch.argmax(logits, dim=2)
        return predictions.byte(), logits

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
            mode='min',
            cooldown=1,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

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
                num_batches=self.config["error_rates_n_batches"],
                predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
            )
            self.log_dict({
                k.replace("*", "_star"): v
                for (k, v) in error_rates.items()
            })
            _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        diacritizer = TorchDiacritizer(self.config, model=self)
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
        predictions = self(batch["src"], batch["lengths"])
        target = batch["target"].contiguous()

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)

        diac_loss = self.criterion(
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
        cls, diacritizer, data_loader, num_batches, predictions_dir
    ):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        all_orig = []
        all_predicted = []
        results = {}
        num_batches_to_take = min(num_batches, len(data_loader))
        for batch in tqdm(
            more_itertools.take(num_batches_to_take, data_loader),
            total=num_batches_to_take,
            desc="Predicting",
            unit="batch",
        ):
            gt_lines = batch["original"]
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
            results = {"DER": 100.0, "WER": 100.0, "DER*": 100.0, "WER*": 100.0}

        return results
