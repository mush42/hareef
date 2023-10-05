# coding: utf-8

import json
import logging
import os
from pathlib import Path
from typing import Any

from .text_encoder import TextEncoder

_LOGGER = logging.getLogger(__package__)


class Config:
    """Model configuration."""

    def __init__(self, config_path: str):
        _LOGGER.info(f"Using config: `{config_path}`")
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = self._load_config()

        self.session_name = self.config["session_name"]
        self.data_dir = Path(self.config["data_directory"])

        self.text_encoder = TextEncoder()
        self.len_input_symbols = len(self.text_encoder.input_symbols)
        self.len_target_symbols = len(self.text_encoder.target_symbols)

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

    def get_inference_config(self):
        token_config = self.text_encoder.dump_tokens()
        return {
            "train_max_length": self["max_len"] - 2,
            **token_config
        }
