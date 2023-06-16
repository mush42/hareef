import argparse
import random
from itertools import repeat

import numpy as np
import torch

from infer import TorchCBHGDiacritizer, OnnxCBHGDiacritizer

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--text", dest="text", type=str, required=True)
    parser.add_argument("--onnx", type=str)

    args = parser.parse_args()

    if args.onnx:
        diacritizer = OnnxCBHGDiacritizer(args.config, onnx_model=args.onnx)
    else:
        diacritizer = TorchCBHGDiacritizer(args.config, load_model=True)

    print("\n".join(diacritizer.diacritize_text(args.text)))


if __name__ == "__main__":
    main()
