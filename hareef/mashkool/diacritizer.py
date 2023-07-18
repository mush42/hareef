# coding: utf-8

import time
from typing import Dict

import numpy as np
import onnxruntime
import torch
from hareef.text_cleaners import diacritics_cleaner, valid_arabic_cleaner
from more_itertools import padded


class Diacritizer:
    """Base diacritizer."""

    def __init__(self, config) -> None:
        self.config = config
        self.text_encoder = self.config.text_encoder
        self.input_pad_id = self.text_encoder.input_pad_id
        self.target_pad_id = self.text_encoder.target_pad_id

    def diacritize_text(self, sentences: list[str]):
        sentences = [diacritics_cleaner(valid_arabic_cleaner(sent)) for sent in sentences]
        char_seqs = [
            self.text_encoder.input_to_sequence(sent)
            for sent in sentences
        ]

        lengths = [len(seq) for seq in char_seqs]

        if len(char_seqs) > 1:
            max_len = max(lengths)
            char_seqs = [
                list(padded(seq, fillvalue=self.input_pad_id, n=max_len))
                for seq in char_seqs
            ]

        char_inputs = np.array(char_seqs,  dtype=np.int64)
        input_lengths = np.array(lengths, dtype=np.int64)

        start_time = time.perf_counter()
        predictions = self.diacritize_batch(char_inputs, input_lengths)
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

    def diacritize_batch(self, char_inputs, input_lengths):
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

    def diacritize_batch(self, char_inputs, input_lengths):
        if self.model is None:
            raise RuntimeError("Called `diacritize_batch` without setting the `model`")
        char_inputs = torch.LongTensor(char_inputs).to(self.device)
        input_lengths = torch.LongTensor(input_lengths).to('cpu')
        outputs = self.model(char_inputs, input_lengths)
        diacritics = outputs["diacritics"]
        predictions = torch.argmax(diacritics, 2)
        return predictions.detach().cpu().numpy()


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model)

    def diacritize_batch(self, char_inputs, input_lengths):
        inputs = {
            "input": char_inputs,
            "input_lengths": input_lengths,
        }
        output = self.session.run(None, inputs)[0]
        predictions = output.argmax(axis=2)
        return predictions
