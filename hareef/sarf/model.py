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
from .modules.conformer import ConformerBlock, ScaledSinusoidalEmbedding



_LOGGER = logging.getLogger(__package__)

class ConvNorm(nn.Module):
    """A 1-dimensional convolutional layer with optional weight normalization.

    This layer wraps a 1D convolutional layer from PyTorch and applies
    optional weight normalization. The layer can be used in a similar way to
    the convolutional layers in PyTorch's `torch.nn` module.

    Args:
        in_channels (int): The number of channels in the input signal.
        out_channels (int): The number of channels in the output signal.
        kernel_size (int, optional): The size of the convolving kernel.
            Defaults to 1.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input.
            If `None`, the padding will be calculated so that the output has
            the same length as the input. Defaults to `None`.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, add bias after convolution. Defaults to `True`.
        w_init_gain (str, optional): The weight initialization function to use.
            Can be either 'linear' or 'relu'. Defaults to 'linear'.
        use_weight_norm (bool, optional): If `True`, apply weight normalization
            to the convolutional weights. Defaults to `False`.

    Shapes:
     - Input: :math:`[N, D, T]`

    - Output: :math:`[N, out_dim, T]` where `out_dim` is the number of output dimensions.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        use_weight_norm=False,
    ):
        super(ConvNorm, self).__init__()  # pylint: disable=super-with-arguments
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_weight_norm = use_weight_norm
        conv_fn = nn.Conv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
        if self.use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal


class SarfModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        hparams = {
            "inp_vocab_size": config.len_input_symbols,
            "hint_vocab_size": config.len_hint_symbols,
            "targ_vocab_size": config.len_target_symbols,
            "input_pad_idx": config.text_encoder.input_pad_id,
            "target_pad_idx": config.text_encoder.target_pad_id,
            **config.config,
            "inference": config.text_encoder.dump_tokens(),
        }
        self.save_hyperparameters(hparams)

        self.config = config
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._build_layers(
            d_model=self.hparams.d_model,
            inp_vocab_size=self.hparams.inp_vocab_size,
            hint_vocab_size=self.hparams.hint_vocab_size,
            targ_vocab_size=self.hparams.targ_vocab_size,
            input_pad_idx=self.hparams.input_pad_idx,
        )

    def _build_layers(
        self,
        d_model,
        inp_vocab_size,
        hint_vocab_size,
        targ_vocab_size,
        input_pad_idx,
    ):
        self.char_emb = nn.Embedding(inp_vocab_size, d_model, padding_idx=input_pad_idx)
        self.diac_emb = nn.Embedding(hint_vocab_size, d_model, padding_idx=input_pad_idx)
        nn.init.uniform_(self.char_emb.weight, -1, 1)
        nn.init.uniform_(self.diac_emb.weight, -1, 1)
        self.pos_enc = ScaledSinusoidalEmbedding(d_model)
        self.dense = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Linear(256, 128),
            nn.Linear(128, d_model)
        )
        self.conv = nn.ModuleList([
            ConvNorm(d_model, d_model, kernel_size=k, use_weight_norm=True)
            for k in (1, 3, 5, 7)
        ])
        self.attn_layers = nn.ModuleList([
            ConformerBlock(
                d_model,
                ffm_dropout=0.2,
                attn_dropout=0.2,
                cgm_dropout=0.3
            )
            for _ in range(3)
        ])
        self.res_layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, targ_vocab_size)

    def forward(self, char_inputs, diac_inputs, lengths):
        length_mask = sequence_mask(
            lengths, char_inputs.size(1)
        ).type_as(char_inputs)

        char_emb = self.char_emb(char_inputs) * 16.00000
        diac_emb = self.diac_emb(diac_inputs) * 32.000
        emb = self.dense(char_emb + diac_emb)
        emb = emb.permute(0, 2, 1)
        mask = length_mask.unsqueeze(1).float()
        for conv in self.conv:
            emb = conv(emb, mask=mask)
        emb = emb.permute(0, 2, 1)

        pos_enc = self.pos_enc(emb)
        b, t, d = emb.size()
        attn_mask = length_mask.bool().unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.expand(b, 1, t, t)
        x = emb
        for attn_layer in self.attn_layers:
            x = attn_layer(x, lengths, attn_mask, pos_enc)

        x = x + emb
        x = self.res_layernorm(x)
        x = self.fc_out(x)
        return x

    def predict(self, char_inputs, diac_inputs, lengths):
        output = self(char_inputs, diac_inputs, lengths)
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
            lr=self.hparams.learning_rate,
            betas=tuple(self.hparams.adam_betas),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
            mode='min',
            cooldown=1,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if ((self.current_epoch + 1) % self.hparams.evaluate_with_error_rates_epochs) == 0:
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
        predictions = self(
            batch["chars"].to(self.device),
            batch["diacs"].to(self.device),
            batch["lengths"].to("cpu")
        )
        target = batch["target"].contiguous()

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)

        diac_loss = self.criterion(
            predictions.to(self.device), targets.to(self.device)
        )
        diac_accuracy = categorical_accuracy(
            predictions.to(self.device),
            targets.to(self.device),
            -100,
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
            predicted, __ = diacritizer.diacritize_text(gt_lines, full_hints=False)
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
