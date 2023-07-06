# coding: utf-8

import functools
import json
import logging

from .dataset import DatasetUtils

_LOGGER = logging.getLogger(__package__)


class Config:
    """Model configuration."""

    def __init__(self, config_path):
        _LOGGER.info(f"Using config: `{config_path}`")

        self.config_path = config_path
        self.config = self._load_config()

    @functools.cached_property
    def data_utils(self):
        return DatasetUtils(self)

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
