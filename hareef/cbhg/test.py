# coding: utf-8

import argparse
import random

import os
from typing import Dict


import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from .config_manager import ConfigManager
from .dataset import load_iterators
from .train.trainer import CBHGTrainer


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DiacritizationTester(CBHGTrainer):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.set_device()

        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id

        self.model = self.config_manager.get_model()

        self.model = self.model.to(self.device)

        self.load_model(model_path=self.config["test_model_path"], load_optimizer=False)
        self.load_diacritizer()
        self.diacritizer.set_model(self.model)

        self.initialize_model()

        self.print_config()

    def run(self):
        self.config_manager.config["load_training_data"] = False
        self.config_manager.config["load_validation_data"] = False
        self.config_manager.config["load_test_data"] = True
        _, test_iterator, _ = load_iterators(self.config_manager)
        tqdm_eval = trange(0, len(test_iterator), leave=True)
        tqdm_error_rates = trange(0, len(test_iterator), leave=True)

        loss, acc = self.evaluate(test_iterator, tqdm_eval)
        error_rates, _ = self.evaluate_with_error_rates(test_iterator, tqdm_error_rates)

        tqdm_eval.close()
        tqdm_error_rates.close()

        WER = error_rates["WER"]
        DER = error_rates["DER"]
        DER1 = error_rates["DER*"]
        WER1 = error_rates["WER*"]

        error_rates = f"DER: {DER}, WER: {WER}, DER*: {DER1}, WER*: {WER1}"

        print(f"global step : {self.global_step}")
        print(f"Evaluate {self.global_step}: accuracy, {acc}, loss: {loss}")
        print(f"WER/DER {self.global_step}: {error_rates}")


def main():
    parser = argparse.ArgumentParser(
        prog="hareef.cbhg.test",
        description="Evaluate the model and print WER/DER statistics. "
    )
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, required=False)

    args = parser.parse_args()

    tester = DiacritizationTester(args.config)
    tester.run()


if __name__ == "__main__":
    main()