# coding: utf-8

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from hareef.utils import find_last_checkpoint

from .config import Config
from .model import HarakatModel

_LOGGER = logging.getLogger("hareef.harakat.export_onnx")
OPSET_VERSION = 18


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.harakat.export_onnx",
        description="Export a model checkpoint to onnx",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = Config(args.config)

    if not args.checkpoint:
        checkpoint_filename, epoch, step = find_last_checkpoint(
            config["logs_root_directory"]
        )
        model = HarakatModel.load_from_checkpoint(
            checkpoint_filename, map_location="cpu", config=config
        )
        _LOGGER.info(f"Using checkpoint from: epoch={epoch} - step={step}")
        _LOGGER.info(f"file: {checkpoint_filename}")
    else:
        model = HarakatModel.load_from_checkpoint(
            args.checkpoint, map_location="cpu", config=config
        )
        _LOGGER.info(f"file: {args.checkpoint}")

    model._infer = model.forward

    def _forward_pass(char_inputs, diac_inputs, input_lengths):
        ret = model._infer(char_inputs, diac_inputs, input_lengths)
        return ret["diacritics"]

    model.forward = _forward_pass
    model.freeze()
    model._jit_is_scripting = True

    inp_vocab_size = config.len_input_symbols
    targ_vocab_size = config.len_target_symbols

    dummy_input_length = 50
    char_inputs = torch.randint(
        low=0, high=inp_vocab_size, size=(1, dummy_input_length), dtype=torch.long
    )
    diac_inputs = torch.randint(
        low=0, high=targ_vocab_size, size=(1, dummy_input_length), dtype=torch.long
    )
    input_lengths = torch.LongTensor([dummy_input_length])
    dummy_input = (char_inputs, diac_inputs, input_lengths)

    # Export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        export_params=True,
        do_constant_folding=True,
        input_names=["char_inputs", "diac_inputs", "input_lengths"],
        output_names=["output"],
        dynamic_axes={
            "char_inputs": {0: "batch_size", 1: "ar_characters"},
            "diac_inputs": {0: "batch_size", 1: "ar_diacritics"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )
    _LOGGER.info(f"Exported model to: {args.output}")


if __name__ == "__main__":
    main()
