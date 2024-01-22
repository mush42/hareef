# coding: utf-8

import statistics
import time
from typing import Dict

import numpy as np
import onnxruntime
import torch
from diacritization_evaluation.util import extract_haraqat
from tqdm import tqdm

from .dataset import load_inference_data


# Sentinel
_missing = object()


class Diacritizer:
    """Base diacritizer."""

    def __init__(self, config) -> None:
        self.config = config
        self.text_encoder = self.config.text_encoder
        self.input_pad_id = self.text_encoder.input_pad_id

    def diacritize_text(self, sentences: list[str], *, full_hints=True, batch_size=_missing):
        if batch_size is _missing:
            batch_size = self.config["batch_size"]
        elif not batch_size:
            batch_size = len(sentences)

        infer_loader = load_inference_data(self.config, sentences, batch_size=batch_size)

        if len(infer_loader) == 1:
            for batch in infer_loader:
                return self._do_diacritize_text(batch, full_hints)

        outputs = []
        infer_times = []
        for batch in tqdm(infer_loader, total=len(infer_loader)):
            sents, infer_time = self._do_diacritize_text(batch, full_hints)
            outputs.extend(sents)
            infer_times.append(infer_time)
        return outputs, statistics.mean(infer_times)

    def _do_diacritize_text(self, batch, full_hints):
        if not full_hints:
            # for evaluation when given fully diacritized sentences
            diac_hints = batch["diacs"]
        else:
            target = batch["target"]
            target[target == -100] = 0
            diac_hints = target
        char_inputs = batch["chars"].cpu().numpy()
        diac_hints = diac_hints.cpu().numpy()
        input_lengths = batch["lengths"].cpu().numpy()
        start_time = time.perf_counter()
        predictions, logits = self.diacritize_batch(char_inputs, diac_hints, input_lengths)
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
        indices, logits = self.model.predict(char_inputs, diac_inputs, input_lengths)
        return (
            indices.detach().cpu().numpy(),
            logits.detach().cpu().numpy(),
        )


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model)

    def diacritize_batch(self, char_inputs, diac_inputs, input_lengths):
        inputs = {
            "char_inputs": char_inputs,
            "diac_inputs": diac_inputs,
            "input_lengths": input_lengths,
        }
        indices, logits = self.session.run(None, inputs)
        return indices, logits
