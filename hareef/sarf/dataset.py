# coding: utf-8

import logging
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from diacritization_evaluation import util
from torch.utils.data import DataLoader, Dataset, BatchSampler, ConcatDataset, WeightedRandomSampler
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
        id = self.list_ids[index]
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
            if (shadda_char in d) and (random.random() <= 0.4):
                d = shadda_char
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


def make_dataset_and_sampler(config, data_split, *, max_sents_per_class=None):
    sampling_config = config["sampling_weights"]
    paths = {
        categ: os.path.join(config.data_dir, categ, f"{data_split}.txt")
        for categ in sampling_config
    }
    diac_datasets = {}
    for ident, path in paths.items():
        with open(path, encoding="utf8") as file:
            lines = file.readlines()
            lines = [line for line in lines if len(line) <= config["max_len"]]
        if max_sents_per_class:
            lines = lines[:max_sents_per_class]
        diac_datasets[ident] = DiacritizationDataset(config, [idx for idx in range(len(lines))], lines)

    dataset_weight = [
        (dataset, sampling_config[ident])
        for ident, dataset in diac_datasets.items()
    ]
    datasets = [dw[0] for dw in dataset_weight]
    weights = torch.concat([
        torch.Tensor([dw[1] for i in range(len(dw[0]))])
        for dw in dataset_weight
    ])
    dataset = ConcatDataset(datasets)
    wr_sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=False
    )
    return dataset, wr_sampler


def load_training_data(config, **loader_parameters):
    dataset, sampler = make_dataset_and_sampler(
        config,
        "train",
        max_sents_per_class=30000
    )
    loader_parameters.setdefault("batch_size", config["batch_size"])
    # loader_parameters.setdefault("shuffle", True)
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
    train_iterator = DataLoader(
        dataset,
        collate_fn=collate_fn,
        sampler=sampler,
        **loader_parameters
    )
    _LOGGER.info(f"Length of training iterator = {len(train_iterator)}")
    return train_iterator


def load_test_data(config, **loader_parameters):
    dataset, sampler = make_dataset_and_sampler(config, "test")
    loader_parameters.setdefault("batch_size", config["batch_size"])
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
    test_iterator = DataLoader(
        dataset,
        collate_fn=collate_fn,
        sampler=sampler,
        **loader_parameters
    )
    _LOGGER.info(f"Length of test iterator = {len(test_iterator)}")
    return test_iterator


def load_validation_data(config, **loader_parameters):
    dataset, sampler = make_dataset_and_sampler(config, "val")

    loader_parameters.setdefault("batch_size", config["batch_size"])
    loader_parameters.setdefault("num_workers", LOADER_NUM_WORKERS)
    valid_iterator = DataLoader(
        dataset,
        collate_fn=collate_fn,
        sampler=sampler,
        **loader_parameters
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
