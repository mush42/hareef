# coding: utf-8

import time
from typing import Dict

import numpy as np
import onnxruntime
import torch
from diacritization_evaluation.util import extract_haraqat
from hareef.text_cleaners import valid_arabic_cleaner
from more_itertools import padded

class Diacritizer:
    """Base diacritizer."""

    def __init__(self, config, *, take_hints=True) -> None:
        self.config = config
        self.take_hints = take_hints
        self.text_encoder = self.config.text_encoder
        self.input_pad_id = self.text_encoder.input_pad_id
        self.target_pad_id = self.text_encoder.target_pad_id

    def diacritize_text(self, sentences: list[str]):
        sentences = [valid_arabic_cleaner(sent) for sent in sentences]

        parsed_sents = [extract_haraqat(sent) for sent in sentences] # -> (original, characters, diacritics)
        char_seqs = [
            self.text_encoder.input_to_sequence(ps[1])
            for ps in parsed_sents
        ]
        diac_seqs = [
            self.text_encoder.target_to_sequence(ps[2])
            for ps in parsed_sents
        ]

        lengths = [len(seq) for seq in char_seqs]
        max_len = max(lengths)

        if len(char_seqs) > 1:
            char_seqs = [
                list(padded(seq, fillvalue=self.input_pad_id, n=max_len))
                for seq in char_seqs
            ]
            diac_seqs = [
                list(padded(seq, fillvalue=self.target_pad_id, n=max_len))
                for seq in diac_seqs
            ]

        char_inputs = np.array(char_seqs,  dtype=np.int64)
        if self.take_hints:
            diac_inputs = np.array(diac_seqs, dtype=np.int64)
        else:
            diac_inputs = np.zeros_like(char_inputs, dtype=np.int64)
        input_lengths = np.array([max_len] * len(lengths), dtype=np.int64)

        start_time = time.perf_counter()
        predictions = self.diacritize_batch(char_inputs, diac_inputs, input_lengths)
        infer_time_ms = (time.perf_counter() - start_time) * 1000

        sentences = []
        for length, src, prediction in zip(input_lengths, char_inputs, predictions):
            src = src[:length + 2]
            prediction = prediction[:length]
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src), list(prediction)
            )
            sentences.append(sentence)

        return sentences, infer_time_ms

    def diacritize_batch(self, char_inputs, diac_inputs, input_lengths):
        raise NotImplementedError


class TorchDiacritizer(Diacritizer):
    """Use `torch` for inference."""

    def __init__(self, *args, model=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if model is not None:
            self.set_model(model)

    def set_model(self, model):
        self.model = model
        self.device = model.device

    def diacritize_batch(self, char_inputs, diac_inputs, input_lengths):
        if self.model is None:
            raise RuntimeError("Called `diacritize_batch` without setting the `model`")
        char_inputs = torch.LongTensor(char_inputs).to(self.device)
        diac_inputs = torch.LongTensor(diac_inputs).to(self.device)
        input_lengths = torch.LongTensor(input_lengths).to('cpu')
        outputs = self.model(char_inputs, diac_inputs, input_lengths)
        diacritics = outputs["diacritics"]
        predictions = torch.argmax(diacritics, 2)
        return predictions.detach().cpu().numpy()


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model)

    def diacritize_batch(self, char_inputs, diac_inputs, input_lengths):
        inputs = {
            "char_inputs": char_inputs,
            #"diac_inputs": diac_inputs,
            "input_lengths": input_lengths,
        }
        output = self.session.run(None, inputs)[0]
        predictions = output.argmax(axis=2)
        return predictions
