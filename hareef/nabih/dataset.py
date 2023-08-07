# coding: utf-8

import logging
import os

import numpy as np
import pandas as pd
import torch
from diacritization_evaluation import util
from torch.utils.data import DataLoader, Dataset
from hareef.utils import clamp


_LOGGER = logging.getLogger(__name__)


class DiacritizationDataset(Dataset):
    """
    The diacritization dataset
    """

    def __init__(self, config, list_ids, data):
        "Initialization"
        self.list_ids = list_ids
        self.data = data
        self.text_encoder = config.text_encoder
        self.config = config

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_ids)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        id = self.list_ids[index]
        if self.config["is_data_preprocessed"]:
            data = self.data.iloc[id]
            inputs = torch.Tensor(self.text_encoder.input_to_sequence(data[1]))
            targets = torch.Tensor(
                self.text_encoder.target_to_sequence(
                    data[2].split(self.config["diacritics_separator"])
                )
            )
            return inputs, targets, data[0]

        data = self.data[id]
        data = self.text_encoder.clean(data)

        text, inputs, diacritics = util.extract_haraqat(data)
        inputs = torch.Tensor(self.text_encoder.input_to_sequence("".join(inputs)))
        diacritics = torch.Tensor(self.text_encoder.target_to_sequence(diacritics))

        return inputs, diacritics, text


def collate_fn(data):
    """
    Padding the input and output sequences
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    src_seqs, trg_seqs, original = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    batch = {
        "original": original,
        "src": src_seqs,
        "target": trg_seqs,
        "lengths": torch.LongTensor(src_lengths),  # src_lengths = trg_lengths
    }
    return batch


def load_training_data(config, **loader_parameters):
    if config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "train.csv")
        train_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config["data_separator"],
            header=None,
        )
        training_set = DiacritizationDataset(config, train_data.index, train_data)
    else:
        path = os.path.join(config.data_dir, "train.txt")
        with open(path, encoding="utf8") as file:
            train_data = file.readlines()
            train_data = [text for text in train_data if len(text) <= config["max_len"]]
        training_set = DiacritizationDataset(
            config, [idx for idx in range(len(train_data))], train_data
        )

    loader_parameters.setdefault("batch_size", config["batch_size"])
    loader_parameters.setdefault("shuffle", True)
    loader_parameters.setdefault("num_workers", os.cpu_count())
    train_iterator = DataLoader(
        training_set, collate_fn=collate_fn, **loader_parameters
    )
    _LOGGER.info(f"Length of training iterator = {len(train_iterator)}")
    return train_iterator


def load_test_data(config, **loader_parameters):
    if config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "test.csv")
        test_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config["data_separator"],
            header=None,
        )
        test_dataset = DiacritizationDataset(config, test_data.index, test_data)
    else:
        test_file_name = "test.txt"
        path = os.path.join(config.data_dir, test_file_name)
        with open(path, encoding="utf8") as file:
            test_data = file.readlines()
        test_data = [text for text in test_data if len(text) <= config["max_len"]]
        test_dataset = DiacritizationDataset(
            config, [idx for idx in range(len(test_data))], test_data
        )

    loader_parameters.setdefault("batch_size", config["batch_size"])
    loader_parameters.setdefault("num_workers", os.cpu_count())
    test_iterator = DataLoader(test_dataset, collate_fn=collate_fn, **loader_parameters)
    _LOGGER.info(f"Length of test iterator = {len(test_iterator)}")
    return test_iterator


def load_validation_data(config, **loader_parameters):
    if config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "val.csv")
        valid_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config["data_separator"],
            header=None,
        )
        valid_dataset = DiacritizationDataset(config, valid_data.index, valid_data)
    else:
        path = os.path.join(config.data_dir, "val.txt")
        with open(path, encoding="utf8") as file:
            valid_data = file.readlines()
        valid_data = [text for text in valid_data if len(text) <= config["max_len"]]
        valid_dataset = DiacritizationDataset(
            config, [idx for idx in range(len(valid_data))], valid_data
        )

    loader_parameters.setdefault("batch_size", config["batch_size"])
    loader_parameters.setdefault("num_workers", os.cpu_count())
    valid_iterator = DataLoader(
        valid_dataset, collate_fn=collate_fn, **loader_parameters
    )
    _LOGGER.info(f"Length of valid iterator = {len(valid_iterator)}")
    return valid_iterator
