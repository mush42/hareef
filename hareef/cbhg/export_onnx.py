# coding: utf-8

import argparse
import random
from pathlib import Path

import torch

from .config_manager import ConfigManager
from .model import CBHGModel
from .util.helpers import find_last_checkpoint


OPSET_VERSION = 15

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        prog="hareef.cbhg.export_onnx",
        description="Export a model checkpoint to onnx"
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = ConfigManager(args.config)

    if not args.checkpoint:
        checkpoint, epoch, step = find_last_checkpoint()
        model = CBHGModel.load_from_checkpoint(checkpoint, map_location='cpu', config=config)
        print(f"Loading model at epoch={epoch} - step: {step}")
    else:
        model = CBHGModel.load_from_checkpoint(args.checkpoint, map_location='cpu', config=config)

    model.freeze()

    inp_vocab_size = config.config["len_input_symbols"]
    dummy_input_length = 50
    input_sequence = torch.randint(
        low=0, high=inp_vocab_size, size=(1, dummy_input_length), dtype=torch.long
    )
    input_sequence_lengths = torch.LongTensor([input_sequence.size(1)])

    dummy_input = (
        input_sequence,
        input_sequence_lengths,
    )

    # Export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        export_params=True,
        do_constant_folding=True,
        input_names=["input", "input_lengths"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "text"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "diacritics"},
        },
    )
    print(f"Exported model to: {args.output}")


if __name__ == "__main__":
    main()
