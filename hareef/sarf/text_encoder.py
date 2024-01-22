# coding: utf-8

import dataclasses
import typing
from collections import OrderedDict
from functools import cached_property
from itertools import chain
from typing import Any, Optional

from hareef.text_cleaners import valid_vocab_char_cleaner
from hareef.constants import (
    ArabicDiacritics,
    ALL_VALID_DIACRITICS,
    ARABIC_LETTERS,
    DIACRITIC_CHARS,
    DIACRITIC_LABELS,
    NUMERAL_CHARS,
    PUNCTUATIONS,
    WORD_SEPARATOR,
)


PAD = "_"
NUM = "#"
INPUT_TOKENS = [
    PAD,
    WORD_SEPARATOR,
    NUM,
    *sorted(chain(PUNCTUATIONS, ARABIC_LETTERS)),
]
TARGET_TOKENS = [PAD, *sorted(ALL_VALID_DIACRITICS)]
HINT_TOKENS = [*TARGET_TOKENS, ArabicDiacritics.SHADDA.value]

MODEL_VOCAB = frozenset(set(INPUT_TOKENS) | set(DIACRITIC_CHARS)) 
NUMERAL_TRANSLATION_TABLE = str.maketrans(
    "".join(NUMERAL_CHARS),
    NUM * len(NUMERAL_CHARS)
)


@dataclasses.dataclass
class TokenConfig:
    pad: str
    num: str
    input_tokens: list[str]
    hint_tokens: list[str]
    target_tokens: list[str]

    def __post_init__(self):
        self.input_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.input_tokens))
        self.hint_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.hint_tokens))
        self.target_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.target_tokens))

    @classmethod
    def default(cls):
        return cls(
            pad=PAD,
            num=NUM,
            input_tokens=INPUT_TOKENS,
            hint_tokens=HINT_TOKENS,
            target_tokens=TARGET_TOKENS,
        )


class TextEncoder:
    """Clean text, prepare input, and convert output."""

    def __init__(self, config: TokenConfig = None):
        self.config = TokenConfig.default() if config is None else config

        self.input_symbols: list[str] = self.config.input_tokens
        self.hint_symbols: list[str] = self.config.hint_tokens
        self.target_symbols: list[str] = self.config.target_tokens

        self.input_symbol_to_id: OrderedDict[str, int] = dict(self.config.input_id_map)
        self.input_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.input_symbol_to_id.items())
        )

        self.hint_symbol_to_id: OrderedDict[str, int] = dict(self.config.hint_id_map)
        self.hint_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.hint_symbol_to_id.items())
        )

        self.target_symbol_to_id: OrderedDict[str, int] = self.config.target_id_map
        self.target_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.target_symbol_to_id.items())
        )

        self.pad = self.config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]

        self.num = self.config.num
        self.input_num_id = self.input_symbol_to_id[self.num]

        self.hint_mask = ArabicDiacritics.NO_DIACRITIC.value
        self.hint_mask_id = self.hint_symbol_to_id[self.hint_mask]

        self.no_diac = ArabicDiacritics.NO_DIACRITIC.value
        self.no_diac_id = self.target_symbol_to_id[self.no_diac]

        self.meta_input_token_ids = {
            self.input_pad_id,
        }
        self.meta_target_token_ids = {
            self.target_pad_id,
        }
        self.shadda_char = ArabicDiacritics.SHADDA.value

    def input_to_sequence(self, text: str) -> list[int]:
        seq = [self.input_symbol_to_id[c] for c in text]
        return seq

    def hint_to_sequence(self, diacs: str) -> list[int]:
        seq = [self.hint_symbol_to_id[c] for c in diacs]
        return seq

    def target_to_sequence(self, diacritics: str) -> list[int]:
        seq = [
            self.target_symbol_to_id.get(s, self.no_diac_id)
            for s in diacritics
        ]
        return seq

    def sequence_to_input(self, sequence: list[int]):
        return [
            self.input_id_to_symbol[symbol_id]
            for symbol_id in sequence
            if (symbol_id not in self.meta_input_token_ids)
        ]

    def sequence_to_hint(self, sequence: list[int]):
        return [
            self.hint_id_to_symbol[symbol_id]
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
        text = text.translate(NUMERAL_TRANSLATION_TABLE)
        return valid_vocab_char_cleaner(text, MODEL_VOCAB)

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
    
    @cached_property
    def target_id_to_label(self):
        ret = {}
        for (idx, symbol) in self.target_id_to_symbol.items():
            ret[idx] = DIACRITIC_LABELS.get(symbol, symbol)
        return OrderedDict(sorted(ret.items()))

    def dump_tokens(self) -> dict[Any, Any]:
        data = {
            "pad": self.config.pad,
            "num": self.config.num,
            "input_id_map": dict(self.input_symbol_to_id),
            "hint_id_map": dict(self.hint_symbol_to_id),
            "target_id_map": dict(self.target_symbol_to_id),
        }
        return {"text_encoder": data}
