# coding: utf-8

import json
import logging
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch

from hareef.text_encoder import HareefTextEncoder, TextEncoderConfig

from .modules.options import AttentionType, LossType, OptimizerType

_LOGGER = logging.getLogger(__package__)


class ConfigManager:
    """Model configuration."""

    def __init__(self, config_path: str):
        _LOGGER.info(f"Using config: `{config_path}`")
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
        encoder_config = TextEncoderConfig(**self.config["text_encoder"])
        self.text_encoder = HareefTextEncoder(encoder_config)
        self.config["len_input_symbols"] = len(self.text_encoder.input_symbols)
        self.config["len_target_symbols"] = len(self.text_encoder.target_symbols)
        self.config["optimizer"] = OptimizerType[self.config["optimizer_type"]]

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def _load_config(self):
        with open(self.config_path, "rb") as model_json:
            _config = json.load(model_json)
        return _config
