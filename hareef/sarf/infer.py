# coding: utf-8

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hareef.utils import find_last_checkpoint
from pysbd import Segmenter


from .config import Config
from .diacritizer import OnnxDiacritizer, TorchDiacritizer
from .model import SarfModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.sarf.infer",
        description="Inference script using Torch or ONNXRuntime",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for inference",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("-f", "--input-file", type=str, required=False)
    parser.add_argument("-o", "--output-file", type=str, required=False)
    parser.add_argument("--text", dest="text", type=str, required=False)
    split_choices = ['none', 'sentence', 'line']
    parser.add_argument("--split-by", type=str, choices=split_choices, default=split_choices[0], help="Split text  (sentence recommended)")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--onnx", type=str, required=False)

    args = parser.parse_args()

    if args.text and args.input_file:
        _LOGGER.error("Passing both text and input-file is ambiguous")
        sys.exit(-1)
    elif args.text:
        input_text = args.text
    elif args.input_file:
        _LOGGER.info(f"Reading text from file: {args.input_file}")
        input_text = Path(args.input_file).read_text(encoding="utf-8")
    else:
        _LOGGER.error("You should provide either --text or --input-file")
        sys.exit(-1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config(args.config)

    if args.onnx:
        _LOGGER.info("Using ONNX backend for inference")
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
                sys.exit(-1)

        model = SarfModel.load_from_checkpoint(args.checkpoint, map_location=args.device, config=config)
        model.freeze()
        diacritizer = TorchDiacritizer(config, model=model)
        _LOGGER.info("Using torch backend for inference")

    if args.split_by == 'sentence':
        _LOGGER.info("Splitting input text by sentence")
        sent_segmenter = Segmenter(language='ar')
        inputs = sent_segmenter.segment(input_text)
    elif args.split_by == 'line':
        _LOGGER.info("Splitting input text by line")
        inputs = input_text.splitlines()
    else:
        inputs = [input_text]
    inputs = [i for i in inputs if i.strip()]

    _LOGGER.info("Running inference on input")
    sents, infer_time = diacritizer.diacritize_text(inputs, args.batch_size)

    if (len(inputs) / args.batch_size) > 1.0:
        _LOGGER.info(f"Average inference time per batch (batch size={args.batch_size}): {infer_time} (ms)")
    else:
        _LOGGER.info(f"Inference time: {infer_time} (ms)")

    output_text = "\n".join(sents)
    if args.output_file:
        _LOGGER.info(f"Writing output to file `{args.output_file}`")
        Path(args.output_file).write_text(
            output_text,
            encoding="utf-8",
            newline="\n"
        )
    else:
        _LOGGER.info("Diacritized sentences:\n" + output_text)


if __name__ == "__main__":
    main()
