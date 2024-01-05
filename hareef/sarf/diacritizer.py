# coding: utf-8

import statistics
import time
from typing import Dict

import numpy as np
import onnxruntime
import torch
from diacritization_evaluation.util import extract_haraqat
from more_itertools import padded, chunked
from tqdm import tqdm


# Sentinel
_missing = object()


class Diacritizer:
    """Base diacritizer."""

    def __init__(self, config) -> None:
        self.config = config
        self.text_encoder = self.config.text_encoder
        self.input_pad_id = self.text_encoder.input_pad_id

    def diacritize_text(self, sentences: list[str], batch_size=_missing):
        if batch_size is _missing:
            batch_size = self.config["batch_size"]
        elif not batch_size:
            batch_size = len(sentences)

        if (len(sentences) / batch_size) <= 1.0:
            return self._do_diacritize_text(sentences)

        batches = list(chunked(sentences, batch_size))
        outputs = []
        infer_times = []
        for batch in tqdm(batches, total=len(batches)):
            sents, infer_time = self._do_diacritize_text(batch)
            outputs.extend(sents)
            infer_times.append(infer_time)
        return outputs, statistics.mean(infer_times)

    def _do_diacritize_text(self, sentences: list[str]):
        sentences = [self.text_encoder.clean(sent) for sent in sentences]

        parsed_sents = [extract_haraqat(sent) for sent in sentences] # -> (original, characters, diacritics)
        char_seqs = [
            self.text_encoder.input_to_sequence(ps[1])
            for ps in parsed_sents
        ]
        char_seqs = list(filter(None, char_seqs))
        lengths = [len(seq) for seq in char_seqs]
        max_len = max(lengths)

        if len(char_seqs) > 1:
            char_seqs = [
                list(padded(seq, fillvalue=self.input_pad_id, n=max_len))
                for seq in char_seqs
            ]

        char_inputs = np.array(char_seqs,  dtype=np.int64)
        input_lengths = np.array(lengths, dtype=np.int64)

        start_time = time.perf_counter()
        predictions, logits = self.diacritize_batch(char_inputs, input_lengths)
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
        indices, logits = self.model.predict(char_inputs, input_lengths)
        return (
            indices.detach().cpu().numpy(),
            logits.detach().cpu().numpy(),
        )


class OnnxDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model)

    def diacritize_batch(self, char_inputs, input_lengths):
        inputs = {
            "char_inputs": char_inputs,
            "input_lengths": input_lengths,
        }
        indices, logits = self.session.run(None, inputs)
        return indices, logits
