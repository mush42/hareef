# coding: utf-8

import logging
import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import more_itertools
import torch
from hareef.learning_rates import cosine_decay_lr_scheduler
from hareef.utils import calculate_error_rates, categorical_accuracy
from lightning.pytorch import LightningModule
from torch import nn, optim

from .dataset import load_test_data, load_validation_data
from .diacritizer import TorchCBHGDiacritizer
from .modules.tacotron_modules import CBHG, Prenet

_LOGGER = logging.getLogger(__package__)


class CBHGModel(LightningModule):
    """
    CBHG model implementation as described in the paper:
     https://ieeexplore.ieee.org/document/9274427
    """

    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.text_encoder = self.config.text_encoder
        self.pad_idx = self.config.text_encoder.input_pad_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}
        self._init_layers(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_vocab_size=self.config["len_target_symbols"],
            use_prenet=self.config["use_prenet"],
            prenet_sizes=self.config["prenet_sizes"],
            cbhg_gru_units=self.config["cbhg_gru_units"],
            cbhg_filters=self.config["cbhg_filters"],
            cbhg_projections=self.config["cbhg_projections"],
            post_cbhg_layers_units=self.config["post_cbhg_layers_units"],
            post_cbhg_use_batch_norm=self.config["post_cbhg_use_batch_norm"],
        )

    def _init_layers(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        use_prenet: bool = True,
        prenet_sizes: List[int] = [512, 256],
        cbhg_gru_units: int = 512,
        cbhg_filters: int = 16,
        cbhg_projections: List[int] = [128, 256],
        post_cbhg_layers_units: List[int] = [256, 256],
        post_cbhg_use_batch_norm: bool = True,
    ):
        """
        Args:
            inp_vocab_size (int): the number of the input symbols
            targ_vocab_size (int): the number of the target symbols (diacritics)
            embedding_dim (int): the embedding  size
            use_prenet (bool): whether to use prenet or not
            prenet_sizes (List[int]): the sizes of the prenet networks
            cbhg_gru_units (int): the number of units of the CBHG GRU, which is the last
            layer of the CBHG Model.
            cbhg_filters (int): number of filters used in the CBHG module
            cbhg_projections: projections used in the CBHG module
        """
        self.use_prenet = use_prenet
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)
        if self.use_prenet:
            self.prenet = Prenet(embedding_dim, prenet_depth=prenet_sizes)

        self.cbhg = CBHG(
            prenet_sizes[-1] if self.use_prenet else embedding_dim,
            cbhg_gru_units,
            K=cbhg_filters,
            projections=cbhg_projections,
        )

        layers = []
        post_cbhg_layers_units = [cbhg_gru_units] + post_cbhg_layers_units

        for i in range(1, len(post_cbhg_layers_units)):
            layers.append(
                nn.LSTM(
                    post_cbhg_layers_units[i - 1] * 2,
                    post_cbhg_layers_units[i],
                    bidirectional=True,
                    batch_first=True,
                )
            )
            if post_cbhg_use_batch_norm:
                layers.append(nn.BatchNorm1d(post_cbhg_layers_units[i] * 2))

        self.post_cbhg_layers = nn.ModuleList(layers)
        self.projections = nn.Linear(post_cbhg_layers_units[-1] * 2, targ_vocab_size)
        self.post_cbhg_layers_units = post_cbhg_layers_units
        self.post_cbhg_use_batch_norm = post_cbhg_use_batch_norm

    def forward(
        self,
        src: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,  # not required in this model
    ):
        """Compute forward propagation"""
        # src = [batch_size, src len]
        # lengths = [batch_size]
        # target = [batch_size, trg len]
        embedding_out = self.embedding(src)
        # embedding_out; [batch_size, src_len, embedding_dim]
        cbhg_input = embedding_out
        if self.use_prenet:
            cbhg_input = self.prenet(embedding_out)
            # cbhg_input = [batch_size, src_len, prenet_sizes[-1]]
        outputs = self.cbhg(cbhg_input, lengths)
        hn = torch.zeros((2, 2, 2))
        cn = torch.zeros((2, 2, 2))
        for i, layer in enumerate(self.post_cbhg_layers):
            if isinstance(layer, nn.BatchNorm1d):
                outputs = layer(outputs.permute(0, 2, 1))
                outputs = outputs.permute(0, 2, 1)
                continue
            if i > 0:
                outputs, (hn, cn) = layer(outputs, (hn, cn))
            else:
                outputs, (hn, cn) = layer(outputs)
        predictions = self.projections(outputs)
        # predictions = [batch_size, src len, targ_vocab_size]
        output = {"diacritics": predictions}
        return output

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        batch["src"] = batch["src"].to(self.device)
        batch["lengths"] = batch["lengths"].to("cpu")
        batch["target"] = batch["target"].to(self.device)
        outputs = self(
            src=batch["src"],
            lengths=batch["lengths"],
            target=batch["target"],
        )
        predictions = outputs["diacritics"].contiguous()
        targets = batch["target"].contiguous()
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        loss = self.criterion(predictions.to(self.device), targets.to(self.device))
        accuracy = categorical_accuracy(
            predictions, targets.to(self.device), self.pad_idx
        )
        self.training_step_outputs.setdefault("loss", []).append(loss)
        self.training_step_outputs.setdefault("accuracy", []).append(accuracy)
        self.log("loss", loss)
        self.log("accuracy", accuracy)

        self.manual_backward(loss)

        gradient_clip_val = self.config.get("gradient_clip_val")
        if gradient_clip_val:
            self.clip_gradients(
                opt, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm="norm"
            )

        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx):
        self.freeze()
        batch["src"] = batch["src"].to(self.device)
        batch["lengths"] = batch["lengths"].to("cpu")
        batch["target"] = batch["target"].to(self.device)
        outputs = self(
            src=batch["src"],
            lengths=batch["lengths"],
            target=batch["target"],
        )
        predictions = outputs["diacritics"].contiguous()
        targets = batch["target"].contiguous()
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        val_loss = self.criterion(predictions.to(self.device), targets.to(self.device))
        val_accuracy = categorical_accuracy(
            predictions, targets.to(self.device), self.pad_idx
        )
        self.val_step_outputs.setdefault("val_loss", []).append(val_loss)
        self.val_step_outputs.setdefault("val_accuracy", []).append(val_accuracy)
        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_accuracy)
        self.unfreeze()
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        self.test_step_outputs.setdefault("test_loss", []).append(metrics["val_loss"])
        self.test_step_outputs.setdefault("test_accuracy", []).append(
            metrics["val_accuracy"]
        )
        return metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = cosine_decay_lr_scheduler(
            optimizer,
            self.config["warmup_steps"],
            self.config["max_epoches"],
            min_lr=0.0,
        )
        return [optimizer], [self.scheduler]

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if (self.current_epoch + 1) % self.config[
            "evaluate_with_error_rates_epoches"
        ] == 0:
            data_loader = load_validation_data(self.config)
            self._evaluate_with_error_rates(data_loader)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        self._evaluate_with_error_rates(data_loader)

    def _log_epoch_metrics(self, metrics):
        for name, values in metrics.items():
            epoch_metric_mean = torch.stack(values).mean()
            self.log(name, epoch_metric_mean)
            values.clear()

    def _evaluate_with_error_rates(self, data_loader):
        predictions_dir = Path(self.trainer.log_dir).joinpath("predictions")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        diacritizer = TorchCBHGDiacritizer(self.config)
        self.freeze()
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
        num_examples = self.config["n_predicted_text_tensorboard"]
        for i, (org, pred) in enumerate(
            more_itertools.take(num_examples, zip(all_orig, all_predicted))
        ):
            self.logger.experiment.add_text(f"eval-text/{i}", f"{org} |->  {pred}")
        return results
