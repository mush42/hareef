# coding: utf-8

import time
from typing import Dict

import numpy as np
import onnxruntime
import torch
from diacritization_evaluation.util import extract_haraqat
from hareef.text_cleaners import diacritics_cleaner, valid_arabic_cleaner


class Diacritizer:
    """Base diacritizer."""

    def __init__(self, config) -> None:
        self.config = config
        self.text_encoder = self.config.text_encoder

    def diacritize_text(self, sentences: list[str]):
        sentences = [diacritics_cleaner(valid_arabic_cleaner(sent)) for sent in sentences]
        pad = self.config.text_encoder.pad
        longest_sent = max(len(sent) for sent in sentences)
        sentences = [sent.ljust(longest_sent, pad) for sent in sentences]
        inputs = np.array(
            [self.text_encoder.input_to_sequence(sent) for sent in sentences],
            dtype=np.int64
        )
        input_lengths = np.array([len(inp) for inp in inputs], dtype=np.int64)
        start_time = time.perf_counter()
        output = self.diacritize_batch(inputs, input_lengths)
        return output, (time.perf_counter() - start_time) * 1000

    def diacritize_batch(self, inputs, input_lengths):
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

    def diacritize_batch(self, inputs, input_lengths):
        if self.model is None:
            raise RuntimeError("Called `diacritize_batch` without setting the `model`")
        inputs = torch.LongTensor(inputs).to(self.device)
        lengths = torch.LongTensor(input_lengths).to('cpu')
        outputs = self.model(inputs, lengths)
        diacritics = outputs["diacritics"]
        predictions = torch.argmax(diacritics, 2)

        sentences = []
        for length, src, prediction in zip(input_lengths, inputs, predictions):
            src = src[:length]
            prediction = prediction[:length]
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)
        return sentences


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model)

    def diacritize_batch(self, inputs, input_lengths):
        output = self.session.run(None, {"input": inputs, "input_lengths": input_lengths})[0]
        predictions = output.argmax(axis=2)

        sentences = []
        for length, src, prediction in zip(input_lengths, inputs, predictions):
            src = src[:length]
            prediction = prediction[:length]
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src), list(prediction)
            )
            sentences.append(sentence)
        return sentences
