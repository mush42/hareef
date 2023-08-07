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
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
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
from .modules.utils import (
    get_dedup_tokens,
    _make_len_mask,
    _generate_square_subsequent_mask,
    PositionalEncoding,
)


_LOGGER = logging.getLogger(__package__)


class ModelType(Enum):
    TRANSFORMER = "transformer"
    AUTOREG_TRANSFORMER = "autoreg_transformer"

    def is_autoregressive(self) -> bool:
        """
        Returns: bool: Whether the model is autoregressive.
        """
        return self in {ModelType.AUTOREG_TRANSFORMER}


class Nabih(LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_pad_idx = self.config.text_encoder.input_pad_id
        self.target_pad_idx = self.config.text_encoder.target_pad_id

        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

    @abstractmethod
    def loss(self, batch, predictions):
        """Calculate loss."""

    @abstractmethod
    def predict(self, inputs):
        """Predict given char inputs."""

    def _set_warmup_lr(
        self,
        warmup_steps: int,
        learning_rate: float,
        step: int,
        optimizer: torch.optim,
    ) -> None:
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = 1.0 - max(warmup_steps - step, 0) / warmup_steps
            lr = learning_rate * warmup_factor
            for g in optimizer.param_groups:
                g["lr"] = lr

    def training_step(self, batch, batch_idx):
        if self.config["warmup_steps"]:
            self._set_warmup_lr(
                self.config["warmup_steps"],
                self.config["learning_rate"],
                self.global_step,
                self.optimizers(),
            )

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
            diacritizer = TorchDiacritizer(self.config, model=self)
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
        outputs = self(batch)

        predictions = outputs["diacritics"].contiguous()
        batch["target"] = batch["target"].contiguous()

        diac_loss = self.loss(batch, predictions)

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)
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
        for batch in tqdm(
            more_itertools.take(num_batches, data_loader),
            total=num_batches,
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


class ForwardTransformer(Nabih):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CTCLoss()

        self._build_layers(
            inp_vocab_size=self.config.len_input_symbols * 3,
            targ_vocab_size=self.config.len_target_symbols,
            embedding_dim=self.config["embedding_dim"],
            d_fft=1024,
            dropout=0.1,
            num_heads=1,
            num_layers=4,
            max_len=self.config["max_len"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        d_fft,
        dropout,
        num_heads,
        num_layers,
        max_len,
        input_pad_idx,
        target_pad_idx,
    ):
        self.embedding = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=input_pad_idx
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=max_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=d_fft,
            dropout=dropout,
            activation="relu",
        )
        encoder_norm = LayerNorm(embedding_dim)
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.fc_out = nn.Linear(embedding_dim, targ_vocab_size)

    def forward(self, batch):
        x = batch["src"].to(self.device)

        x = x.transpose(0, 1)  # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return {"diacritics": x}

    def predict(self, inputs):
        return self({"src": inputs})

    def loss(self, batch, predictions):
        predictions = predictions.transpose(0, 1).log_softmax(2)
        lengths = batch["lengths"].to("cpu")
        return self.criterion(
            predictions,
            batch["target"].to(self.device),
            lengths,
            lengths,
        )


class AutoregressiveTransformer(Nabih):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_pad_idx)

        self.input_eos_id = self.config.text_encoder.input_eos_id
        self.target_eos_id = self.config.text_encoder.target_eos_id

        self._build_layers(
            inp_vocab_size=self.config.len_input_symbols * 3,
            targ_vocab_size=self.config.len_target_symbols,
            embedding_dim=self.config["embedding_dim"],
            d_fft=1024,
            dropout=0.1,
            num_heads=1,
            num_encoder_layers=4,
            num_decoder_layers=4,
            max_len=self.config["max_len"],
            input_pad_idx=self.config.text_encoder.input_pad_id,
            target_pad_idx=self.config.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        d_fft,
        dropout,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len,
        input_pad_idx,
        target_pad_idx,
    ):
        self.encoder = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=input_pad_idx
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=max_len)
        self.decoder = nn.Embedding(
            targ_vocab_size, embedding_dim, padding_idx=target_pad_idx
        )
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_fft,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(embedding_dim, targ_vocab_size)

    def forward(self, batch):
        src = batch["src"].to(self.device)
        trg = batch["target"].to(self.device)

        src = src.transpose(0, 1)  # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = _generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = _make_len_mask(src).to(trg.device)
        trg_pad_mask = _make_len_mask(trg).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(
            src,
            trg,
            src_mask=None,
            tgt_mask=trg_mask,
            memory_mask=None,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return {"diacritics": output}

    def predict(self, input, max_len: int = 100):
        batch_size = input.size(0)
        input = input.transpose(0, 1)  # shape: [T, N]
        src_pad_mask = _make_len_mask(input).to(input.device)
        with torch.no_grad():
            input = self.encoder(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input, src_key_padding_mask=src_pad_mask)
            out_indices = torch.zeros_like(input).float()
            for i in range(max_len):
                tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(
                    output,
                    input,
                    memory_key_padding_mask=src_pad_mask,
                    tgt_mask=tgt_mask,
                )
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self.target_eos_id, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        return {"diacritics": out_indices}

    def loss(self, batch, predictions):
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)
        return self.criterion(predictions.to(self.device), targets.to(self.device))


NabihModel = ForwardTransformer
# NabihModel = AutoregressiveTransformer
