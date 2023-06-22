# coding: utf-8

import argparse
import random
from itertools import repeat

import numpy as np
import torch

from ..config_manager import ConfigManager
from ..model import CBHGModel
from ..util.helpers import find_last_checkpoint
from .diacritizer import OnnxCBHGDiacritizer, TorchCBHGDiacritizer


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        prog="hareef.cbhg.infer",
        description="Inference script using Torch or ONNXRuntime"
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--text", dest="text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--onnx", type=str, required=False)

    args = parser.parse_args()

    config = ConfigManager(args.config)

    if args.onnx:
        diacritizer = OnnxCBHGDiacritizer(config, onnx_model=args.onnx)
    else:
        if not args.checkpoint:
            try:
                checkpoint_filename, epoch, step = find_last_checkpoint()
                print(f"Using checkpoint from: epoch={epoch} - step={step}")
                print(f"file: {checkpoint_filename}")
                args.checkpoint = checkpoint_filename
            except:
                print("Failed to obtain the path to the last checkpoint")
                raise

        model = CBHGModel.load_from_checkpoint(args.checkpoint, config=config)
        model.freeze()
        diacritizer = TorchCBHGDiacritizer(config, model=model)

    sents, infer_time = diacritizer.diacritize_text(args.text)
    print(f"Inference time (ms): {infer_time}")
    print("\n".join(sents))


if __name__ == "__main__":
    main()
