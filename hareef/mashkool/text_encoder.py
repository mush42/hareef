# coding: utf-8

import dataclasses
import typing
from itertools import chain
from typing import Any, Optional

from hareef.text_cleaners import valid_arabic_cleaner
from hareef.constants import (
    ALL_VALID_DIACRITICS,
    ARABIC_LETTERS,
    PUNCTUATIONS,
    WORD_SEPARATOR,
)


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
INPUT_TOKENS = [
    PAD,
    SOS,
    EOS,
    WORD_SEPARATOR,
    *sorted(chain(PUNCTUATIONS, ARABIC_LETTERS)),
]
TARGET_TOKENS = [PAD, SOS, EOS, *sorted(ALL_VALID_DIACRITICS)]
INPUT_ID_MAP = {char: idx for idx, char in enumerate(INPUT_TOKENS)}
TARGET_ID_MAP = {char: idx for idx, char in enumerate(TARGET_TOKENS)}
DEFAULT_TOKEN_MAP = {
    "pad": PAD,
    "sos": SOS,
    "eos": EOS,
    "input_id_map": INPUT_ID_MAP,
    "target_id_map": TARGET_ID_MAP,
}


@dataclasses.dataclass(frozen=True, slots=True)
class TokenConfig:
    pad: str
    sos: str
    eos: str
    input_id_map: dict[str, int]
    target_id_map: dict[str, int]

    @classmethod
    def default(cls):
        return cls(**DEFAULT_TOKEN_MAP)


class TextEncoder:
    """Clean text, prepare input, and convert output."""

    def __init__(self, config: TokenConfig = None):
        self.config = TokenConfig.default() if config is None else config

        self.input_symbols: list[str] = list(self.config.input_id_map.keys())
        self.target_symbols: list[str] = list(self.config.target_id_map.keys())

        self.input_symbol_to_id: dict[str, int] = dict(self.config.input_id_map)
        self.input_id_to_symbol: dict[int, str] = {
            id: char for char, id in self.input_symbol_to_id.items()
        }

        self.target_symbol_to_id: dict[str, int] = dict(self.config.target_id_map)
        self.target_id_to_symbol: dict[int, str] = {
            id: char for char, id in self.target_symbol_to_id.items()
        }

        self.pad = self.config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]

        self.sos = self.config.sos
        self.input_sos_id = self.input_symbol_to_id[self.sos]
        self.target_sos_id = self.target_symbol_to_id[self.sos]

        self.eos = self.config.eos
        self.input_eos_id = self.input_symbol_to_id[self.eos]
        self.target_eos_id = self.target_symbol_to_id[self.eos]

        self.meta_input_token_ids = {
            self.input_pad_id,
            self.input_sos_id,
            self.input_eos_id,
        }
        self.meta_target_token_ids = {
            self.target_pad_id,
            self.target_sos_id,
            self.target_eos_id,
        }

    def input_to_sequence(self, text: str) -> list[int]:
        seq = [self.input_symbol_to_id[s] for s in text if s != self.pad]
        return [self.input_sos_id, *seq, self.input_eos_id]

    def target_to_sequence(self, diacritics: str) -> list[int]:
        seq = [self.target_symbol_to_id[s] for s in diacritics if s != self.pad]
        return [self.target_sos_id, *seq, self.target_eos_id]

    def sequence_to_input(self, sequence: list[int]):
        return [
            self.input_id_to_symbol[symbol_id]
            for symbol_id in sequence
            if (symbol_id not in self.meta_input_token_ids)
        ]

    def sequence_to_target(self, sequence: list[int]):
        return [
            self.target_id_to_symbol[symbol_id]
            for symbol_id in sequence
            if (symbol_id not in self.meta_target_token_ids)
        ]

    def clean(self, text):
        return valid_arabic_cleaner(text)

    def combine_text_and_diacritics(self, input_ids: list[int], output_ids: list[int]):
        """
        Combines the  input text with its corresponding  diacritics
        Args:
            input_ids: a list of ids representing the input text
            output_ids: a list of ids representing the output text
        Returns:
        text: the text after merging the inputs text representation with the output
        representation
        """
        return "".join(
            letter + diac
            for (letter, diac) in zip(
                self.sequence_to_input(input_ids), self.sequence_to_target(output_ids)
            )
        )

    def dump_tokens(self) -> dict[Any, Any]:
        data = {
            "pad": self.config.pad,
            "sos": self.config.sos,
            "eos": self.config.eos,
            "input_id_map": dict(self.input_symbol_to_id),
            "target_id_map": dict(self.target_symbol_to_id),
        }
        return {"text_encoder": data}
