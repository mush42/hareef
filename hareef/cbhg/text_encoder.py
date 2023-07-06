# coding: utf-8

import dataclasses
import typing
from typing import Any, Optional

from hareef.text_cleaners import valid_arabic_cleaner
from hareef.constants import ALL_POSSIBLE_DIACRITICS, ARABIC_LETTERS, PUNCTUATIONS, WORD_SEPARATOR


PAD = "<PAD>"
INPUT_TOKENS = [PAD, WORD_SEPARATOR, *sorted(PUNCTUATIONS), *sorted(ARABIC_LETTERS)]
TARGET_TOKENS = [PAD, *sorted(ALL_POSSIBLE_DIACRITICS)]
INPUT_ID_MAP = {char: idx for idx, char in enumerate(INPUT_TOKENS)}
TARGET_ID_MAP = {char: idx for idx, char in enumerate(TARGET_TOKENS)}
DEFAULT_TOKEN_MAP = {
    "pad": PAD,
    "input_id_map": INPUT_ID_MAP,
    "target_id_map": TARGET_ID_MAP
}


@dataclasses.dataclass(frozen=True, slots=True)
class CBHGTokenConfig:
    pad: str
    input_id_map: dict[str, int]
    target_id_map: dict[str, int]

    @classmethod
    def default(cls):
        return cls(**DEFAULT_TOKEN_MAP)



class CBHGTextEncoder:
    """Clean text, prepare input, and convert output."""

    def __init__(self, config: CBHGTokenConfig=None):
        self.config = CBHGTokenConfig.default() if config is None else config

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

        self.meta_input_token_ids = {self.input_pad_id}
        self.meta_target_token_ids = {self.target_pad_id,}

    def input_to_sequence(self, text: str) -> list[int]:
        seq = [self.input_symbol_to_id[s] for s in text if s != self.pad]
        return seq
    
    def target_to_sequence(self, text: str) -> list[int]:
        seq = [self.target_symbol_to_id[s] for s in text if s != self.pad]
        return seq

    def sequence_to_input(self, sequence: list[int]):
        return [
            self.input_id_to_symbol[symbol_id]
            for symbol_id in sequence
            if (symbol_id in self.input_id_to_symbol)
            and (symbol_id not in self.meta_input_token_ids)
        ]

    def sequence_to_target(self, sequence: list[int]):
        return [
            self.target_id_to_symbol[symbol_id]
            for symbol_id in sequence
            if (symbol_id in self.target_id_to_symbol)
            and (symbol_id not in self.meta_target_token_ids)
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
            "input_id_map": dict(self.input_symbol_to_id),
            "target_id_map": dict(self.target_symbol_to_id),
        }
        return {"text_encoder": data}
