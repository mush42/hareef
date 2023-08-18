# coding: utf-8

import os
from torch.utils.data import DataLoader

from .data_utils import DatasetUtils
from .dataloader import DataRetriever


def load_training_data(config):
    return DataLoader(
        DataRetriever('train', config.data_utils),
        batch_size=config["train"]["batch-size"],
        shuffle=True,
        num_workers=os.cpu_count()
    )


def load_val_data(config):
    return DataLoader(
        DataRetriever('val', config.data_utils),
        batch_size=min(config["train"]["batch-size"], 32),
        shuffle=False,
        num_workers=os.cpu_count()
    )


def load_test_data(config, *, is_test=False):
    return DataLoader(
        DataRetriever('test', config.data_utils, is_test=is_test),
        batch_size=min(config["train"]["batch-size"], 32),
        shuffle=False,
        num_workers=os.cpu_count()
    )
