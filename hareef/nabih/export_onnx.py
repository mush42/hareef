# coding: utf-8

import argparse
import logging
import os
import random
from pathlib import Path

import more_itertools
import numpy as np
import torch
from hareef.utils import categorical_accuracy, find_last_checkpoint, format_as_table

from .config import Config
from .dataset import load_training_data, load_validation_data
from .model import NabihModel

_LOGGER = logging.getLogger("hareef.nabih.export_onnx")

DEFAULT_OPSET_VERSION = 17


class QuantDataLoader:
    def __init__(self, loader, batch_size, is_fix_dims, max_len):
        self.loader = loader
        self.batch_size = batch_size
        self.is_fix_dims = is_fix_dims
        self.max_len = max_len

    def __iter__(self):
        if self.is_fix_dims:
            return self.make_iterator_for_fixed_dims(self.loader)
        return self.make_iterator(self.loader)

    def make_iterator(self, loader):
        yield from (
            ((batch["src"], batch["lengths"]), batch["target"])
            for batch in iter(loader)
        )

    def make_iterator_for_fixed_dims(self, loader):
        max_len = self.max_len
        for batch in iter(loader):
            inputs = batch["src"].long()
            pad = torch.zeros(inputs.shape[0], max_len - inputs.shape[1]).long()
            inputs = torch.cat([inputs, pad], dim=1)
            lengths = batch["lengths"].long()
            inputs = inputs.unbind(0)
            lengths = lengths.unbind(0)
            targets = batch["target"].unbind(0)
            yield from zip(
                zip(
                    inputs,
                    (l.unsqueeze(0) for l in lengths)
                ),
                targets
            )



class QuantCategoricalAccuracy:
    def __init__(self, config):
        self.target_pad_idx = config.text_encoder.target_pad_id
        self.preds = []
        self.targets = []

    def update(self, preds, labels):
        preds = preds[1]
        self.preds.append(preds.reshape(-1, preds.shape[-1]))
        self.targets.append(labels[0].reshape(-1))

    def reset(self):
        self.preds.clear()
        self.targets.clear()

    def result(self):
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        predictions = torch.Tensor(preds)
        targets = torch.Tensor(targets)
        accuracy = categorical_accuracy(
            predictions.to("cpu"),
            targets.to("cpu"),
            self.target_pad_idx,
            device="cpu"
        )
        return accuracy.item()


def to_onnx(model, config, output_filename, opset=DEFAULT_OPSET_VERSION, fix_dims=False):
    inp_vocab_size = config.len_input_symbols
    model._infer = model.forward
    if fix_dims:
        _LOGGER.info("Fixing input and output dims. Batched input will not be available for the exported model.")
        def _forward_pass(inputs, lengths):
            inputs = inputs.unsqueeze(0)
            output = model._infer(inputs, lengths)
            logits = torch.softmax(output, dim=2)
            predictions = torch.argmax(logits, dim=2)
            max_logits = torch.max(logits, dim=2).values
            return (
                predictions.squeeze(0).byte(),
                max_logits.squeeze(0)
            )

        dummy_input_length = config["max_len"]
        char_inputs = torch.randint(
            low=0, high=inp_vocab_size, size=(dummy_input_length,), dtype=torch.long
        )
        input_lengths = torch.LongTensor([dummy_input_length])
        dummy_input = (char_inputs, input_lengths)
        dynamic_axes = None
    else:
        def _forward_pass(inputs, lengths):
            output = model._infer(inputs, lengths)
            logits = torch.softmax(output, dim=2)
            predictions = torch.argmax(logits, dim=2)
            max_logits = torch.max(logits, dim=2).values
            return predictions.byte(), max_logits

        dummy_input_length = 50
        char_inputs = torch.randint(
            low=0, high=inp_vocab_size, size=(1, dummy_input_length), dtype=torch.long
        )
        input_lengths = torch.LongTensor([dummy_input_length])
        dummy_input = (char_inputs, input_lengths)
        dynamic_axes={
            "char_inputs": {0: "batch_size", 1: "ar_characters"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        }

    model.forward = _forward_pass

    # Export
    _LOGGER.info(f"Using ONNX OPSET version {opset}")
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(output_filename),
        verbose=False,
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        input_names=["char_inputs", "input_lengths"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    return output_filename


def simplify(input_model_path):
    import onnx
    from onnxsim import simplify

    _LOGGER.info("Simplifying model using ONNX simplifier...")
    model = onnx.load(input_model_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    output_path = os.fspath(Path(input_model_path).with_suffix(".simp.onnx"))
    onnx.save(
        model_simp,
        output_path
    )
    return output_path


def quantize(input_model_path, config, fix_dims):
    from neural_compressor import quantization
    from neural_compressor.config import PostTrainingQuantConfig

    _LOGGER.info("Quantizing model using neural compressor...")

    train_loader = load_training_data(config, num_workers=0)
    val_loader = load_validation_data(config, num_workers=0)
    calib_dataloader = QuantDataLoader(
        more_itertools.take(1000, train_loader),
        batch_size=config["batch_size"],
        is_fix_dims=fix_dims,
        max_len=config["max_len"]
    )
    eval_dataloader = QuantDataLoader(
        val_loader,
        batch_size=config["batch_size"],
        is_fix_dims=fix_dims,
        max_len=config["max_len"]
    )

    q_conf = PostTrainingQuantConfig(domain='nlp')
    q_accuracy = QuantCategoricalAccuracy(config)
    q_model = quantization.fit(
        model=input_model_path,
        conf=q_conf,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader,
        eval_metric=q_accuracy
    )
    output_path = os.fspath(Path(input_model_path).with_suffix(".quant.onnx"))
    q_model.save_model_to_file(output_path)
    return output_path


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="hareef.nabih.export_onnx",
        description="Export a model checkpoint to onnx",
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--quant", action="store_true", help="Quantize the onnx model.")
    parser.add_argument("--opset", dest="opset_version", type=int, default=DEFAULT_OPSET_VERSION, help="ONNX OPSET version")
    parser.add_argument("--fix-dims", action="store_true", help="Fix input and output dims to maintain compatibility with certain ONNX runtimes")
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

    model.freeze()
    model._jit_is_scripting = True

    saved_files = []

    onnx_model = to_onnx(
        model=model,
        config=config,
        output_filename=args.output,
        opset=args.opset_version,
        fix_dims=args.fix_dims,
    )
    _LOGGER.info(f"Exported model to: {args.output}")
    saved_files.append(("original", args.output))

    if args.simp:
        simplified_model_path = simplify(onnx_model)
        _LOGGER.info(f"Saved simplified model to: {simplified_model_path}")
        saved_files.append(("simp", simplified_model_path))

    if args.quant:
        quant_model_path = quantize(simplified_model_path, config=config, fix_dims=args.fix_dims)
        _LOGGER.info(f"Saved quantized model to: {quant_model_path}")
        saved_files.append(("quant", quant_model_path))

    cols = [
        ("", ["  size (MB)"]),
        *[(name, [os.path.getsize(path) // 1e+6]) for name, path in saved_files]
    ]
    _LOGGER.info("\n\nModel size:\n" + format_as_table(*cols))


if __name__ == "__main__":
    main()
