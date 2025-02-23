# coding: utf-8

import logging
import os
import random
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import more_itertools
import numpy as np
import transformers
import torch
from diacritization_evaluation.util import extract_haraqat
from lightning.pytorch import LightningModule
from torch import nn, optim
from tqdm import tqdm

from hareef.utils import (calculate_error_rates, categorical_accuracy,
                          format_error_rates_as_table, sequence_mask)

from .dataset import load_test_data, load_validation_data
from .diacritizer import TorchDiacritizer
from .modules import TransformerEncoder, TransformerDecoder, POSPredictor, ScaledSinusoidalEmbedding


_LOGGER = logging.getLogger(__package__)




class SarfModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        hparams = {
            "inp_vocab_size": config.len_input_symbols,
            "hint_vocab_size": config.len_hint_symbols,
            "targ_vocab_size": config.len_target_symbols,
            "num_pos_tags": config.text_encoder.num_pos_tags,
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
        self.pos_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._build_layers(
            d_model=self.hparams.d_model,
            inp_vocab_size=self.hparams.inp_vocab_size,
            hint_vocab_size=self.hparams.hint_vocab_size,
            targ_vocab_size=self.hparams.targ_vocab_size,
            num_pos_tags=self.hparams.num_pos_tags,
            input_pad_idx=self.hparams.input_pad_idx,
        )

    def _build_layers(
        self,
        d_model,
        inp_vocab_size,
        hint_vocab_size,
        targ_vocab_size,
        num_pos_tags,
        input_pad_idx,
    ):
        self.char_emb = nn.Embedding(inp_vocab_size, d_model, padding_idx=input_pad_idx)
        self.diac_emb = nn.Embedding(
            hint_vocab_size, d_model, padding_idx=input_pad_idx
        )
        nn.init.uniform_(self.char_emb.weight, -1, 1)
        nn.init.uniform_(self.diac_emb.weight, -1, 1)
        self.position_enc = ScaledSinusoidalEmbedding(d_model, 2000)
        self.encoder = TransformerEncoder(dim=d_model)
        pos_encoder_dim = 64
        self.pos_encoder = POSPredictor(dim=d_model, output_dim=pos_encoder_dim, num_pos_tags=num_pos_tags)
        output_dim = d_model + pos_encoder_dim
        self.decoder = TransformerDecoder(dim=output_dim)
        self.diac_fc_out = nn.Linear(output_dim, targ_vocab_size)

    def forward(self, char_inputs, diac_inputs, lengths):
        length_mask = sequence_mask(lengths, char_inputs.size(1)).type_as(char_inputs)
        padding_mask = ~length_mask.squeeze(1).bool()

        x = self.char_emb(char_inputs) * 16.00000
        x = x + self.diac_emb(diac_inputs) * 32.000

        position_enc = self.position_enc(x)
        x = x + position_enc
        enc_out = self.encoder(x, padding_mask) 

        pos_emb, pos_logits = self.pos_encoder(x, padding_mask)
        x = torch.concat([x, pos_emb], dim=-1)

        x = self.decoder(x, padding_mask)

        diac_logits = self.diac_fc_out(x)

        return diac_logits, pos_logits

    @torch.inference_mode()
    def predict(self, char_inputs, diac_inputs, lengths):
        output, __ = self(char_inputs, diac_inputs, lengths)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.99),
        )
        max_steps = 2_00_000
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=max_steps,
            last_epoch=getattr("self", "ckpt_loaded_epoch", -1)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if (
            (self.current_epoch + 1) % self.hparams.evaluate_with_error_rates_epochs
        ) == 0:
            data_loader = load_validation_data(self.config)
            diacritizer = TorchDiacritizer(self.config, model=self)
            error_rates = self.evaluate_with_error_rates(
                diacritizer,
                data_loader=data_loader,
                num_batches=self.config["error_rates_n_batches"],
                predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
            )
            self.log_dict(
                {k.replace("*", "_star"): v for (k, v) in error_rates.items()}
            )
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
        predictions, pos_predictions = self(
            batch["chars"].to(self.device),
            batch["diacs"].to(self.device),
            batch["lengths"].to("cpu"),
        )
        target = batch["target"].contiguous()
        pos_target = batch["char2pos"].contiguous()

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)
        pos_predictions = pos_predictions.view(-1, pos_predictions.shape[-1])
        pos_targets = pos_target.view(-1)

        diac_loss = self.criterion(predictions.to(self.device), targets.to(self.device))
        pos_loss = self.pos_criterion(
            pos_predictions.to(self.device), pos_targets.to(self.device)
        )
        diac_accuracy = categorical_accuracy(
            predictions.to(self.device),
            targets.to(self.device),
            -100,
            device=self.device,
        )
        pos_accuracy = categorical_accuracy(
            pos_predictions.to(self.device),
            pos_targets.to(self.device),
            -100,
            device=self.device,
        )

        loss = (2.0 * pos_loss) + (8.0 * diac_loss)
        return {
            "loss": loss,
            "diac_loss": diac_loss,
            "pos_loss": pos_loss,
            "diac_accuracy": diac_accuracy,
            "pos_accuracy": pos_accuracy,
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
