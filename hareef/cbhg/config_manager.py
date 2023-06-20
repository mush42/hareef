# coding: utf-8

import json
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch

from hareef.text_encoder import HareefTextEncoder, TextEncoderConfig

from .model import CBHGModel
from .modules.options import AttentionType, LossType, OptimizerType


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        model_kind = self.model_kind = "cbhg"
        self.config: Dict[str, Any] = self._load_config()
        self.session_name = ".".join(
            [
                self.config["data_type"],
                self.config["session_name"],
                f"{model_kind}",
            ]
        )

        self.data_dir = Path(
            os.path.join(self.config["data_directory"], self.config["data_type"])
        )
        self.base_dir = Path(
            os.path.join(self.config["log_directory"], self.session_name)
        )
        self.log_dir = Path(os.path.join(self.base_dir, "logs"))
        self.prediction_dir = Path(os.path.join(self.base_dir, "predictions"))
        self.plot_dir = Path(os.path.join(self.base_dir, "plots"))
        self.models_dir = Path(os.path.join(self.base_dir, "models"))
        encoder_config = TextEncoderConfig(**self.config["text_encoder"])
        self.text_encoder = HareefTextEncoder(encoder_config)
        self.config["len_input_symbols"] = len(self.text_encoder.input_symbols)
        self.config["len_target_symbols"] = len(self.text_encoder.target_symbols)
        self.config["optimizer"] = OptimizerType[self.config["optimizer_type"]]

    def _load_config(self):
        with open(self.config_path, "rb") as model_json:
            _config = json.load(model_json)
        return _config

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * " "
        print(tab + "-", key_name, ":", values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(
                    dictionary[key], key_name=key, level=recursion_level
                )

    def print_config(self):
        print("\nCONFIGURATION", self.session_name)
        self._print_dictionary(self.config)

    def dump_config(self):
        _config = {}
        for key, val in self.config.items():
            if isinstance(val, Enum):
                _config[key] = val.name
            else:
                _config[key] = val
        with open(self.base_dir / "config.json", "w") as model_json:
            json.dump(_config, model_json, indent=2)
        # needed only for inference
        infer_config = {
            "max_len": _config["max_len"],
            "text_encoder": _config["text_encoder"],
        }
        with open(self.base_dir / "inference-config.json", "w", encoding="utf-8") as infer_json:
            json.dump(infer_config, infer_json, ensure_ascii=False, indent=2)

    def create_remove_dirs(
        self,
        clear_dir: bool = False,
        clear_logs: bool = False,
        clear_weights: bool = False,
        clear_all: bool = False,
    ):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.prediction_dir.mkdir(exist_ok=True)
        if clear_dir:
            delete = input(f"Delete {self.log_dir} AND {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.models_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f"Delete {self.log_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f"Delete {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.models_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def get_last_model_path(self):
        """
        Given a checkpoint, get the last save model name
        Args:
        checkpoint (str): the path where models are saved
        """
        models = os.listdir(self.models_dir)
        models = [model for model in models if model[-3:] == ".pt"]
        if len(models) == 0:
            return None
        _max = max(int(m.split(".")[0].split("-")[0]) for m in models)
        model_name = f"{_max}-snapshot.pt"
        last_model_path = os.path.join(self.models_dir, model_name)

        return last_model_path

    def load_model(self, model_path: str = None):
        """
        loading a model from path
        """
        model = self.get_model()

        with open(self.base_dir / f"{self.model_kind}_network.txt", "w") as file:
            file.write(str(model))

        if model_path is None:
            last_model_path = self.get_last_model_path()
            if last_model_path is None:
                return model, 1
        else:
            last_model_path = model_path

        saved_model = torch.load(last_model_path)
        out = model.load_state_dict(saved_model["model_state_dict"])
        print(out)
        global_step = saved_model["global_step"] + 1
        return model, global_step

    def get_model(self):
        return self.get_cbhg()

    def get_cbhg(self):
        model = CBHGModel(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_vocab_size=self.config["len_target_symbols"],
            use_prenet=self.config["use_prenet"],
            prenet_sizes=self.config["prenet_sizes"],
            cbhg_gru_units=self.config["cbhg_gru_units"],
            cbhg_filters=self.config["cbhg_filters"],
            cbhg_projections=self.config["cbhg_projections"],
            post_cbhg_layers_units=self.config["post_cbhg_layers_units"],
            post_cbhg_use_batch_norm=self.config["post_cbhg_use_batch_norm"],
        )

        return model

    def get_loss_type(self):
        try:
            loss_type = LossType[self.config["loss_type"]]
        except:
            raise Exception(f"The loss type is not correct {self.config['loss_type']}")
        return loss_type
