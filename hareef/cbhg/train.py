# coding: utf-8

import argparse
import logging
import random

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .config_manager import ConfigManager
from .dataset import load_iterators
from .model import CBHGModel


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.cbhg.training",
        description="Training script for hareef.cbhg model."
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    choices = ['gpu', 'cpu']
    parser.add_argument("--accelerator", type=str, default=choices[0], choices=choices)    
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--test", action="store_true", help="Run the test after training")
    parser.add_argument("--debug", action="store_true", help="Use fast dev mode of lightning")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = ConfigManager(args.config)

    precision = "16-mixed" if config.config["use_mixed_precision"] else 32
    checkpoint_save_callback = ModelCheckpoint(
        every_n_train_steps=config["model_save_steps"],
        every_n_epochs=config["model_save_epoches"]
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        check_val_every_n_epoch=config.config["evaluate_epoches"],
        callbacks = [checkpoint_save_callback,],
        max_epochs=config.config["max_epoches"],
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=config.get("gradient_clip_val"),
        gradient_clip_algorithm="norm",
        fast_dev_run=args.debug,
        log_every_n_steps=10
    )

    model = CBHGModel(config)

    if args.test:
        config.config["load_test_data"] = True
        __, test_loader, __ = load_iterators(config)
        trainer.test(model, test_loader)
    else:
        train_loader, __, val_loader = load_iterators(config)
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
