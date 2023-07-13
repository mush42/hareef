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

    def diacritize_text(self, text: str):
        text = diacritics_cleaner(valid_arabic_cleaner(text))
        inputs_seq = self.text_encoder.input_to_sequence(text)
        start_time = time.perf_counter()
        output = self.diacritize_batch(inputs_seq)
        return output, (time.perf_counter() - start_time) * 1000

    def diacritize_batch(self, batch):
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

    def diacritize_batch(self, input_seq):
        if self.model is None:
            raise RuntimeError("Called `diacritize_batch` without setting the `model`")
        inputs = torch.LongTensor([input_seq]).to(self.device)
        outputs = self.model(inputs.to(self.device))
        diacritics = outputs["diacritics"]
        predictions = torch.argmax(diacritics, 2)

        sentences = []
        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)
        return sentences


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model

    def diacritize_batch(self, batch):
        session = onnxruntime.InferenceSession(self.onnx_model)
        inputs = np.expand_dims(batch, axis=0).astype(np.int64)
        output = session.run(None, {"input": inputs})[0]
        predictions = output.argmax(axis=2)

        sentences = []
        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_diacritics(
                list(src), list(prediction)
            )
            sentences.append(sentence)

        return sentences
