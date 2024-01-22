# coding: utf-8

import logging
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from diacritization_evaluation import util
from torch.utils.data import DataLoader, Dataset
from hareef.utils import clamp


LOADER_NUM_WORKERS = 0

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

        text, inputs, diacs = util.extract_haraqat(data)
        inputs = torch.Tensor(self.text_encoder.input_to_sequence("".join(inputs)))
        diacritics = torch.Tensor(self.text_encoder.target_to_sequence(diacs))
        hints = self._generate_diac_hints(diacs)

        return inputs, hints, diacritics, text

    def _generate_diac_hints(self, diacs, no_hints=False):
        if no_hints:
            nh = torch.zeros(len(diacs)) + self.text_encoder.hint_mask_id
            return nh.long()
        p = np.random.uniform()
        shadda_char = self.text_encoder.shadda_char
        augmented_diacs = []
        for d in diacs:
            if (shadda_char in d) and (np.random.beta(7, 14) <= 0.6):
                augmented_diacs.append(shadda_char)            
                continue
            augmented_diacs.append(d)
        diac_seq = torch.LongTensor(self.text_encoder.hint_to_sequence(augmented_diacs))
        diac_mask = torch.bernoulli(torch.full(diac_seq.shape, p))
        diac_seq = diac_seq * diac_mask.long()
        diac_seq[diac_seq == 0] = self.text_encoder.hint_mask_id
        return diac_seq

def collate_fn(data, enforce_sorted=True):
    """
    Padding the input and output sequences
    """

    def merge(sequences, pad_idx=0):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        if pad_idx != 0:
            padded_seqs = padded_seqs + pad_idx
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    if enforce_sorted:
        data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    src_seqs, hint_seqs, trg_seqs, original = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    diac_seqs, __ = merge(hint_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs, pad_idx=-100)

    batch = {
        "original": original,
        "chars": src_seqs,
        "diacs": diac_seqs,
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
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
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
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
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
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
    valid_iterator = DataLoader(
        valid_dataset, collate_fn=collate_fn, **loader_parameters
    )
    _LOGGER.info(f"Length of valid iterator = {len(valid_iterator)}")
    return valid_iterator


def load_inference_data(config, sents, batch_size=None, **loader_parameters):
    infer_dataset = DiacritizationDataset(
        config,
        list(range(len(sents))),
        sents
    )

    loader_parameters.setdefault(
        "batch_size", batch_size or config["batch_size"]
    )
    loader_parameters.setdefault("num_workers", 0)
    infer_iterator = DataLoader(
        infer_dataset,
        collate_fn=partial(collate_fn, enforce_sorted=False),
        **loader_parameters
    )
    return infer_iterator
