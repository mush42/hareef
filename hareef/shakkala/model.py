# coding: utf-8

import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch
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
from .modules.attention import Attention
from .modules.k_lstm import K_LSTM
from .modules import markov_lstm



_LOGGER = logging.getLogger(__package__)


class ShakkalaModel(LightningModule):
    """ """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = self.config.text_encoder
        self.input_pad_idx = self.text_encoder.input_pad_id
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.diac_criterion = nn.CrossEntropyLoss(
            ignore_index=self.input_pad_idx
        )
        self.hints_criterion = nn.CrossEntropyLoss(
            ignore_index=self.input_pad_idx
        )
        self._build_layers(
            inp_vocab_size=config.len_input_symbols,
            targ_vocab_size=config.len_target_symbols,
            embedding_dim=config["embedding_dim"],
            lstm_dims=config["lstm_dims"],
            lstm_dropout=config["lstm_dropout"],
            input_pad_idx=self.text_encoder.input_pad_id,
            target_pad_idx=self.text_encoder.target_pad_id,
        )

    def _build_layers(
        self,
        inp_vocab_size,
        targ_vocab_size,
        embedding_dim,
        lstm_dims,
        lstm_dropout,
        input_pad_idx,
        target_pad_idx,
    ):
        (
            (diac_enc_dim, diac_enc_layers),
            (lstm2_h, lstm2_n_layers),
            (diac_dec_dim, diac_dec_layers),
        ) = lstm_dims
        self.diac_embedding = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=input_pad_idx
        )
        self.hints_embedding = nn.Embedding(
            targ_vocab_size, embedding_dim, padding_idx=target_pad_idx
        )
        self.diac_encoder = K_LSTM(
            input_size=embedding_dim,
            hidden_size=diac_enc_dim,
            num_layers=diac_enc_layers,
            batch_first=True,
            bidirectional=True,
            recurrent_activation="hard_sigmoid",
            vertical_dropout=lstm_dropout,
            recurrent_dropout=lstm_dropout,
            return_states=False,
        )
        self.hints_encoder = K_LSTM(
            input_size=embedding_dim,
            hidden_size=diac_enc_dim * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            recurrent_activation="hard_sigmoid",
            vertical_dropout=lstm_dropout,
            recurrent_dropout=lstm_dropout,
            return_states=False,
        )
        self.diac_enc_batchnorm = nn.BatchNorm1d(
            num_features=diac_enc_dim * 2,
            eps=1e-3,
            momentum=0.1,
        )
        self.diac_decoder = markov_lstm.LSTM(
            cell_class=markov_lstm.LSTMCell,
            input_size=self.diac_enc_batchnorm.num_features,
            hidden_size=diac_dec_dim * 2,
            output_size=diac_dec_dim,
            num_layers=diac_dec_layers,
            k_cells=3,
            dropout_prob=0.2
        )
        self.projections = nn.Linear(diac_dec_dim * 2, targ_vocab_size)

    def forward(self, src, hints=None):
        if hints is None:
            hints = torch.zeros_like(src)
        diac_embedding_out = self.diac_embedding(src)
        hints_embedding_out = self.hints_embedding(hints)
        diac_enc_out = self.diac_encoder(diac_embedding_out).tanh()
        hints_enc_out = self.hints_encoder(hints_embedding_out).tanh()
        enc_out = diac_enc_out + hints_enc_out
        diac_enc_batchnorm_out = self.diac_enc_batchnorm(enc_out.transpose(1, 2))
        outs = self.diac_decoder(diac_enc_batchnorm_out.transpose(1, 2), 5.0, is_training=self.training)
        diac_dec_out = outs[3].tanh()
        predictions = self.projections(diac_dec_out).log_softmax(dim=1)
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
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["weight_decay"],
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
        batch["hints"] = batch["hints"].to(self.device)
        outputs = self(batch["src"], batch["hints"])
        predictions = outputs["diacritics"].contiguous()
        targets = batch["target"].contiguous()
        hints = batch["hints"].contiguous()
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        hints = hints.view(-1)
        diac_loss = self.diac_criterion(predictions.to(self.device), targets.to(self.device))
        # Create and apply predictions mask based on hints
        mask = hints.ne(0).bool()
        masked_hints = hints[mask]
        masked_predictions = predictions[mask]
        hints_loss = self.hints_criterion(masked_predictions.to(self.device), masked_hints.to(self.device))
        diac_accuracy = categorical_accuracy(predictions, targets.to(self.device), self.input_pad_idx)
        hints_accuracy = categorical_accuracy(predictions, hints.to(self.device), self.input_pad_idx)
        loss = (10. * diac_loss) + (3. * hints_loss)
        return {
            "loss": loss,
            "diac_loss": diac_loss,
            "hints_loss": hints_loss,
            "diac_accuracy": diac_accuracy,
            "hints_accuracy": hints_accuracy
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
