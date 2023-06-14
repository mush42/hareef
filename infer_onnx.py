import numpy as np
import onnxruntime
from config_manager import ConfigManager


class Diacritizer:
    def __init__(self, config_path: str) -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id

    def diacritize_text(self, text: str):
        seq = self.text_encoder.input_to_sequence(text)
        return self.diacritize_batch(seq)

    def diacritize_batch(self, batch):
        raise NotImplementedError


class CBHGDiacritizer(Diacritizer):
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
