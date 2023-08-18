# coding: utf-8

import argparse
import functools
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from hareef.utils import find_last_checkpoint
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin

from ..config import Config
from ..dataset import load_training_data, load_val_data, load_test_data
from .model import DiacritizerD3

_LOGGER = logging.getLogger("hareef.cbhg.train")


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.ddiac.d2.train",
        description="Training script for hareef.deep_-diac.d2 model.",
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
    parser.add_argument(
        '--from-scratch', action="store_true", help="Train D3 without requireing  a pretrained D2 checkpoint."
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config(args.config)

    logs_root_directory = Path(config["paths"]["logs"])
    logs_root_directory.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Logs directory: {logs_root_directory}")

    checkpoint_save_callback = ModelCheckpoint(
        every_n_train_steps=config["train"]["model-save-steps"],
        every_n_epochs=config["train"]["model-save-epochs"],
    )
    stopping_delta = config["train"]["stopping-delta"]
    stopping_patience = config["train"]["stopping-patience"]
    loss_early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=stopping_delta, patience=stopping_patience, mode="min", strict=True
    )
    plugins = []
    if config["train"]["mixed-precision"]:
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
        check_val_every_n_epoch=config["train"]["evaluate-every-epochs"],
        callbacks=[
            loss_early_stop_callback,
            checkpoint_save_callback,
        ],
        plugins=plugins,
        max_epochs=config["train"]["max-epoches"],
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

    _LOGGER.info("Initializing model...")
    model = DiacritizerD3(config)

    if args.test:
        _LOGGER.info("Testing loop starting...")
        test_loader = load_test_data(config)
        trainer.test(model, test_loader)
    else:
        if not os.path.isfile(config["paths"]["d2-checkpoint"] or ""):
            if  args.from_scratch:
                _LOGGER.warn(
                    "You passed `--from_scratch` when starting D3 training.\n"
                    "This is not recommended\n"
                    "For more information, please take a look at the paper."
                )
            else:
                raise RuntimeError(
                    "D3 model training requires a pretrained D2 checkpoint.\n"
                    "Please add the path to a pretrained D2 checkpoint to the model config\n"
                    "or pass `--from-scratch` to train the model from scratch"
                )
        else:
            pretrained_dict = torch.load(config["paths"]["d2-checkpoint"], map_location=T.device(model.device))["state_dict"]
            model_state_dict = model.state_dict()
            for i, (name, params) in enumerate(pretrained_dict.items()):
                if name in model_state_dict and i < 41:
                    model_state_dict[name].copy_(params)
            model.load_state_dict(model_state_dict)
            if config["train"]["freeze-base"]:
                _LOGGER.info("> Freezing Model parameters...")
                for params in model.parameters():
                    params.requires_grad = False 
                for params in model.lstm_decoder.parameters():
                    params.requires_grad = True 
                for params in model.classifier.parameters():
                    params.requires_grad = True 
        train_loader, val_loader = load_training_data(config), load_val_data(config)
        inference_config_path = logs_root_directory.joinpath("inference-config.json")
        with open(inference_config_path, "w", encoding="utf-8", newline="\n") as file:
            json.dump(
                "\n", file, ensure_ascii=False, indent=2
            )
        _LOGGER.info(f"Writing inference config to file: `{inference_config_path}`")
        _LOGGER.info("Training loop starting...")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
