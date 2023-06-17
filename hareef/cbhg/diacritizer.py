# coding: utf-8

import time
from typing import Dict

import numpy as np
import onnxruntime
import torch

from .config_manager import ConfigManager
from .util.constants import HARAQAT
from .util.text_cleaners import valid_arabic_cleaners

HARAQAT_TRANS_DICT = {ord(c): None for c in HARAQAT}


class Diacritizer:
    def __init__(self, config_path: str) -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id

    def diacritize_text(self, text: str):
        text = valid_arabic_cleaners(text).translate(HARAQAT_TRANS_DICT)
        seq = [self.start_symbol_id, *self.text_encoder.input_to_sequence(text)]
        start_time = time.perf_counter()
        output = self.diacritize_batch(seq)
        print(f"Inference time (ms): {(time.perf_counter() - start_time) * 1000}")
        return output

    def diacritize_batch(self, batch):
        raise NotImplementedError


class TorchCBHGDiacritizer(Diacritizer):
    def __init__(self, *args, load_model: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load_model = load_model

    def set_model(self, model):
        self.model = model

    def diacritize_batch(self, batch):
        if self.config.get("device"):
            self.device = self.config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.load_model:
            self.model, self.global_step = self.config_manager.load_model()
            self.model = self.model.to(self.device)

        self.model.eval()

        batch = torch.LongTensor([batch]).to(self.device)
        inputs = batch.data
        lengths = torch.tensor([batch.shape[1]], dtype=torch.int64).to("cpu")
        outputs = self.model(inputs.to(self.device), lengths)
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences


class OnnxCBHGDiacritizer(Diacritizer):
    def __init__(self, *args, onnx_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_model = onnx_model

    def diacritize_batch(self, batch):
        session = onnxruntime.InferenceSession(self.onnx_model)
        inputs = np.expand_dims(batch, axis=0).astype(np.int64)
        input_lengths = np.array([inputs.shape[1]], dtype=np.int64)
        output = session.run(None, {"input": inputs, "input_lengths": input_lengths})[0]
        predictions = output.argmax(axis=2)

        sentences = []
        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src), list(prediction)
            )
            sentences.append(sentence)

        return sentences
