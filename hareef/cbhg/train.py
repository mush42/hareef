# coding: utf-8


import argparse
import random

import numpy as np
import torch

from .trainer import CBHGTrainer

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
    parser.add_argument(
        "--reset_dir",
        dest="clear_dir",
        action="store_true",
        help="deletes everything under this config's folder.",
    )

    args = parser.parse_args()

    trainer = CBHGTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
