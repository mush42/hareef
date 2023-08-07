# coding: utf-8

import argparse
import logging
import os
import random
from pathlib import Path

import more_itertools
import numpy as np
import torch
from hareef.utils import find_last_checkpoint


from .config import Config
from .dataset import load_validation_data
from .model import NabihModel

_LOGGER = logging.getLogger("hareef.nabih.export_onnx")
OPSET_VERSION = 18



def simplify(model_path):
    import onnx
    from onnxsim import simplify

    _LOGGER.info("Simplifying model using ONNX simplifier...")
    model = onnx.load(model_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    output_path = os.fspath(Path(model_path).with_suffix(".simp.onnx"))
    onnx.save(
        model_simp,
        output_path
    )
    _LOGGER.info(f"Saved simplified model to: {output_path}")




def quantize(input_model_path, data_iter):
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.quantization.quantize  import quantize_static
    from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod

    class CDataReader(CalibrationDataReader):
        def __init__(self, data_iter):
            super().__init__()
            self.data_iter = iter(data_iter)

        def get_next(self):
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return
            return {
                "char_inputs": batch["src"].detach().cpu().numpy(),
                "input_lengths": batch["lengths"].detach().cpu().numpy()
            }

    preprocess_output_path = output_model_path=os.fspath(Path(input_model_path).with_suffix(".quant.onnx"))
    _LOGGER.info("Preprocessing model for quantization")
    quant_pre_process(
        input_model_path=input_model_path,
        output_model_path=preprocess_output_path,
        auto_merge=True,
    )
    _LOGGER.info("Quantizing model..")
    quantize_static(
        model_input=preprocess_output_path,
        model_output=os.fspath(Path(preprocess_output_path).with_suffix(".static.onnx")),
        calibration_data_reader=CDataReader(data_iter),
        optimize_model=False,
    )


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.nabih.export_onnx",
        description="Export a model checkpoint to onnx",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--quant", action="store_true", help="Quantize the onnx model.")
    parser.add_argument("--simp", action="store_true", help="Simplify the onnx model using onnx-simplifier tool")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = Config(args.config)

    if not args.checkpoint:
        checkpoint_filename, epoch, step = find_last_checkpoint(
            config["logs_root_directory"]
        )
        model = NabihModel.load_from_checkpoint(
            checkpoint_filename, map_location="cpu", config=config
        )
        _LOGGER.info(f"Using checkpoint from: epoch={epoch} - step={step}")
        _LOGGER.info(f"file: {checkpoint_filename}")
    else:
        model = NabihModel.load_from_checkpoint(
            args.checkpoint, map_location="cpu", config=config
        )
        _LOGGER.info(f"file: {args.checkpoint}")

    model._infer = model.forward

    def _forward_pass(char_inputs, lengths):
        ret = model._infer({"src": char_inputs, "lengths": lengths})
        return ret["diacritics"]

    model.forward = _forward_pass
    model.freeze()
    model._jit_is_scripting = True

    inp_vocab_size = config.len_input_symbols
    dummy_input_length = 50
    char_inputs = torch.randint(
        low=0, high=inp_vocab_size, size=(1, dummy_input_length), dtype=torch.long
    )
    input_lengths = torch.LongTensor([dummy_input_length])
    dummy_input = (char_inputs, input_lengths)

    # Export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        export_params=True,
        do_constant_folding=True,
        input_names=["char_inputs", "input_lengths"],
        output_names=["output"],
        dynamic_axes={
            "char_inputs": {0: "batch_size", 1: "ar_characters"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )
    _LOGGER.info(f"Exported model to: {args.output}")
    if args.simp:
        simplify(args.output)
    if args.quant:
        data_loader = load_validation_data(config, num_workers=0)
        data_iter = more_itertools.take(10, iter(data_loader))
        quantize(args.output, data_iter)

if __name__ == "__main__":
    main()
