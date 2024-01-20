# coding: utf-8

import argparse
import logging
import random
import sys
from tempfile import TemporaryDirectory

import more_itertools
import numpy as np
import torch
from hareef.utils import find_last_checkpoint, format_error_rates_as_table, generate_confusion_matrix
from tqdm import tqdm

from .config import Config
from .diacritizer import TorchDiacritizer, OnnxDiacritizer
from .dataset import load_test_data, load_validation_data
from .model import SarfModel

_LOGGER = logging.getLogger(__package__)


def error_rates(diacritizer, data_loader, num_batches):
    _LOGGER.info("Calculating DER/WER statistics...")
    try:
        with TemporaryDirectory() as predictions_dir:
            error_rates = SarfModel.evaluate_with_error_rates(diacritizer, data_loader, num_batches=num_batches, predictions_dir=predictions_dir)
    except:
        _LOGGER.error("Failed to calculate DER/WER statistics", exc_info=True)
        sys.exit(1)

    _LOGGER.info("Error Rates:\n" + format_error_rates_as_table(error_rates))



def confusion_matrix(diacritizer, data_loader, num_batches, plot, fig_save_path):
    _LOGGER.info("Generating and plotting  model confusion matrix...")

    for unwanted_logger in ["matplotlib", "PIL"]:
        logging.getLogger(unwanted_logger).setLevel(logging.WARNING)

    gt_lines = []
    pred_lines = []
    total = min(num_batches, len(data_loader))
    for batch in tqdm(more_itertools.take(num_batches, data_loader), total=total, desc="Predicting", unit="batch"):
        batch_lines = batch["original"]
        gt_lines.extend(batch_lines)
        predicted, __ = diacritizer.diacritize_text(batch_lines)
        pred_lines.extend(predicted)

    generate_confusion_matrix(gt_lines, pred_lines, plot, fig_save_path)


def main():
    logging.basicConfig(level=logging.DEBUG)


    parser = argparse.ArgumentParser(
        prog="hareef.sarf.metrics",
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
    parser.add_argument("--num-batches", type=int, required=False)
    parser.add_argument("--checkpoint", type=str, required=False, help="Use torch for inference")
    parser.add_argument("--onnx", type=str, required=False, help="Use onnx for inference (provides significant speedups)")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    err_rates_parser = subparsers.add_parser("rates", help="Calculate DER/WER statistics")
    conmat_parser = subparsers.add_parser("conmat", help="Calculate and plot confusion matrix")


    conmat_parser.add_argument("--plot", action="store_true", help="Show confusion matrix plot")
    conmat_parser.add_argument("--fig-save-path", type=str, help="Save confusion matrix plot to the specified path")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config(args.config)

    if args.checkpoint and args.onnx:
        _LOGGER.error("Only one of `checkpoint` or `onnx` is allowed")
        sys.exit(1)

    if args.onnx:
        _LOGGER.info(f"Using ONNX model from: {args.onnx}")
        diacritizer = OnnxDiacritizer(config, onnx_model=args.onnx)
    else:
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

        _LOGGER.info(f"Using checkpoint from: {args.checkpoint}")
        device = args.device if args.device != 'gpu' else 'cuda'
        model = SarfModel.load_from_checkpoint(
            args.checkpoint, map_location=device, config=config
        )
        model.freeze()
        diacritizer = TorchDiacritizer(config, model=model)

    data_loader = (
        load_test_data(config, num_workers=0)
        if args.datasplit == "test"
        else load_validation_data(config, num_workers=0)
    )

    if args.subcommand == 'rates':
        num_batches = args.num_batches or len(data_loader)
        return error_rates(diacritizer, data_loader, num_batches=num_batches)
    elif args.subcommand == 'conmat':
        num_batches = args.num_batches or 20
        return confusion_matrix(diacritizer, data_loader, num_batches=num_batches, plot=args.plot, fig_save_path=args.fig_save_path)
    else:
        _LOGGER.error("Please pass a metric name to view")
        sys.exit(1)


if __name__ == "__main__":
    main()
