# coding: utf-8

import dataclasses
import typing
from typing import Optional

from .text_cleaners import valid_arabic_cleaner
from .constants import ALL_POSSIBLE_DIACRITICS, ARABIC_LETTERS, PUNCTUATIONS



@dataclasses.dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    bos: str
    eos: str
    pad: str
    input_id_map: dict[str, int]
    target_id_map: dict[str, int]



class HareefTextEncoder:
    """Clean text, prepare input, and convert output."""

    def __init__(self, config: TextEncoderConfig):
        self.config = config

        self.input_symbols: list[str] = list(self.config.input_id_map.keys())
        self.target_symbols: list[str] = list(self.config.target_id_map.keys())

        self.input_symbol_to_id: dict[str, int] = dict(self.config.input_id_map)
        self.input_id_to_symbol: dict[int, str] = {id: char for char, id in self.input_symbol_to_id.items()}

        self.target_symbol_to_id: dict[str, int] = dict(self.config.target_id_map)
        self.target_id_to_symbol: dict[int, str] = {id: char for char, id in self.target_symbol_to_id.items()}

        self.pad = self.config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]
        self.start_symbol_id = self.target_symbol_to_id[self.config.bos]

    def input_to_sequence(self, text: str) -> list[int]:
        sequence = [self.input_symbol_to_id[s] for s in text if s !=  self.pad]
        return sequence

    def target_to_sequence(self, text: str) -> list[int]:
        sequence = [self.target_symbol_to_id[s] for s in text if s !=  self.pad]
        return sequence

    def sequence_to_input(self, sequence: list[int]):
        return [
            self.input_id_to_symbol[symbol]
            for symbol in sequence
            if (symbol in self.input_id_to_symbol) and (symbol !=  self.input_pad_id)
        ]

    def sequence_to_target(self, sequence: list[int]):
        return [
            self.target_id_to_symbol[symbol]
            for symbol in sequence
            if (symbol in self.target_id_to_symbol) and (symbol != self.target_pad_id)
        ]

    def clean(self, text):
        return valid_arabic_cleaner(text)

    def combine_text_and_haraqat(self, input_ids: list[int], output_ids: list[int]):
        """
        Combines the  input text with its corresponding  haraqat
        Args:
            inputs: a list of ids representing the input text
            outputs: a list of ids representing the output text
        Returns:
        text: the text after merging the inputs text representation with the output
        representation
        """
        output = ""
        for i, input_id in enumerate(input_ids):
            if input_id == self.input_pad_id:
                break
            output += self.input_id_to_symbol[input_id]
            output += self.target_id_to_symbol[output_ids[i]]
        return output
