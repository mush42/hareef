# coding: utf-8

import argparse
import logging
import random
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from hareef.utils import find_last_checkpoint, format_error_rates_as_table

from .config import Config
from .dataset import load_test_data, load_validation_data
from .model import ShakkalaModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.shakkala.error_rates",
        description="Calculate DER/WER diacritization error rates",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device used for inference",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Dataset split to use (val or test)",
    )
    parser.add_argument("--checkpoint", type=str, required=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config(args.config)

    if not args.checkpoint:
        try:
            checkpoint_filename, epoch, step = find_last_checkpoint(
                config["logs_root_directory"]
            )
            _LOGGER.info(f"Using checkpoint from: epoch={epoch} - step={step}")
            _LOGGER.info(f"file: {checkpoint_filename}")
            args.checkpoint = checkpoint_filename
        except:
            _LOGGER.exception(
                "Failed to obtain the path to the last checkpoint", exc_info=True
            )
            sys.exit(1)

    model = ShakkalaModel.load_from_checkpoint(
        args.checkpoint, map_location=args.device, config=config
    )
    model.freeze()

    iterator = (
        load_test_data(config)
        if args.datasplit == "test"
        else load_validation_data(config)
    )
    try:
        with TemporaryDirectory() as predictions_dir:
            error_rates = model.evaluate_with_error_rates(iterator, predictions_dir)
    except:
        _LOGGER.error("Failed to calculate DER/WER statistics", exc_info=True)
        sys.exit(1)

    _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))


if __name__ == "__main__":
    main()
