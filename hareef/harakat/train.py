# coding: utf-8

import argparse
import functools
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from hareef.utils import find_last_checkpoint
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin

from .config import Config
from .dataset import load_test_data, load_training_data, load_validation_data
from .model import HarakatModel

_LOGGER = logging.getLogger("hareef.harakat.train")


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.harakat.train",
        description="Training script for hareef.harakat model.",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    choices = ["gpu", "cpu"]
    parser.add_argument("--accelerator", type=str, default=choices[0], choices=choices)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--continue-from", type=str, help="Checkpoint to continue training from"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the test after training"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use fast dev mode of lightning"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config(args.config)
    logs_root_directory = Path(config["logs_root_directory"])
    logs_root_directory.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Logs directory: {logs_root_directory}")

    model = HarakatModel(config)

    checkpoint_save_callbacks = []
    if config["model_save_steps"]:
        checkpoint_save_callbacks.append(ModelCheckpoint(every_n_train_steps=config["model_save_steps"]))
    if config["model_save_epochs"]:
        checkpoint_save_callbacks.append(ModelCheckpoint(every_n_epochs=config["model_save_epochs"]))

    loss_early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, mode="min", strict=True
    )
    plugins = []
    if config["use_mixed_precision"]:
        _LOGGER.info("Configuring automatic mixed precision")
        mp = (
            MixedPrecisionPlugin(
                "16-mixed", device="cuda", scaler=torch.cuda.amp.GradScaler()
            )
            if args.accelerator == "gpu"
            else MixedPrecisionPlugin("bf16-mixed", device="cpu")
        )
        plugins = [
            mp,
        ]
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        check_val_every_n_epoch=config["evaluate_epochs"],
        callbacks=[
            loss_early_stop_callback,
            *checkpoint_save_callbacks,
        ],
        plugins=plugins,
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.debug,
        log_every_n_steps=10,
        default_root_dir=logs_root_directory,
    )

    if args.continue_from:
        if args.continue_from == "last":
            checkpoint_filename, epoch, step = find_last_checkpoint(
                config["logs_root_directory"]
            )
            args.continue_from = checkpoint_filename
            _LOGGER.info(
                f"Automatically using checkpoint last checkpoint from: epoch={epoch} - step={step}"
            )
        _LOGGER.info(f"Continueing training from checkpoint: {args.continue_from}")
        trainer.ckpt_path = args.continue_from

    if args.test:
        _LOGGER.info("Testing loop starting...")
        test_loader = load_test_data(config)
        trainer.test(model, test_loader)
    else:
        train_loader, val_loader = load_training_data(config), load_validation_data(
            config
        )
        inference_config_path = logs_root_directory.joinpath("inference-config.json")
        with open(inference_config_path, "w", encoding="utf-8", newline="\n") as file:
            json.dump(
                config.text_encoder.dump_tokens(), file, ensure_ascii=False, indent=2
            )
        _LOGGER.info(f"Writing inference config to file: `{inference_config_path}`")
        _LOGGER.info("Training loop starting...")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
