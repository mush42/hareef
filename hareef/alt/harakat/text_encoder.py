# coding: utf-8

import dataclasses
import typing
from collections import OrderedDict
from functools import cached_property
from itertools import chain
from typing import Any, Optional

from hareef.text_cleaners import valid_arabic_cleaner
from hareef.constants import (
    DIACRITIC_CHARS,
    ALL_VALID_DIACRITICS,
    ARABIC_LETTERS,
    PUNCTUATIONS,
    WORD_SEPARATOR,
    DIACRITIC_LABELS,
    WORD_DELIMITERS,
    ARABIC_VOWELS,
    ArabicDiacritics,
)


PAD = "_"
SOS = "^"
EOS = "$"
INPUT_TOKENS = [
    PAD,
    SOS,
    EOS,
    WORD_SEPARATOR,
    *sorted(chain(PUNCTUATIONS, ARABIC_LETTERS)),
]
TARGET_TOKENS = [PAD, SOS, EOS, *sorted(ALL_VALID_DIACRITICS)]


@dataclasses.dataclass
class TokenConfig:
    pad: str
    sos: str
    eos: str
    input_tokens: list[str]
    target_tokens: list[str]

    def __post_init__(self):
        self.input_id_map: OrderedDict[str, int] = OrderedDict(
            (char, idx) for idx, char in enumerate(self.input_tokens)
        )
        self.target_id_map: OrderedDict[str, int] = OrderedDict(
            (char, idx) for idx, char in enumerate(self.target_tokens)
        )

    @classmethod
    def default(cls):
        return cls(
            pad=PAD,
            sos=SOS,
            eos=EOS,
            input_tokens=INPUT_TOKENS,
            target_tokens=TARGET_TOKENS,
        )


class TextEncoder:
    """Clean text, prepare input, and convert output."""

    def __init__(self, config: TokenConfig = None):
        self.config = TokenConfig.default() if config is None else config

        self.input_symbols: list[str] = self.config.input_tokens
        self.target_symbols: list[str] = self.config.target_tokens

        self.input_symbol_to_id: OrderedDict[str, int] = dict(self.config.input_id_map)
        self.input_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.input_symbol_to_id.items())
        )

        self.target_symbol_to_id: OrderedDict[str, int] = self.config.target_id_map
        self.target_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.target_symbol_to_id.items())
        )

        self.pad = self.config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]

        self.sos = self.config.sos
        self.input_sos_id = self.input_symbol_to_id[self.sos]
        self.target_sos_id = self.target_symbol_to_id[self.sos]

        self.eos = self.config.eos
        self.input_eos_id = self.input_symbol_to_id[self.eos]
        self.target_eos_id = self.target_symbol_to_id[self.eos]

        self.meta_tokens = {self.pad, self.sos, self.eos,}

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
        seq = [self.input_symbol_to_id[s] for s in text]
        return [self.input_sos_id, *seq, self.input_eos_id]

    def target_to_sequence(self, diacritics: str) -> list[int]:
        seq = [self.target_symbol_to_id[s] for s in diacritics]
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

    def combine_text_and_diacritics(
        self,
        input_ids: list[int],
        output_ids: list[int],
        logits: list[int],
        apply_golden_rule: Optional[bool] = False,
        golden_rule_threshold: Optional[float] = 1.0,
    ):
        """
        Combines the  input text with its corresponding  diacritics
        Args:
            input_ids: a list of ids representing the input text
            output_ids: a list of ids representing the output text
            logits: a list of probabilities  representing the model's certainty about output diacritic
            apply_golden_rule: whether to replace CE diacritics with sukoon if the model is uncertain
            golden_rule_threshold: the cut-off probability below  which we apply the golden rule
        Returns:
        text: the text after merging the inputs text representation with the output
        representation
        """
        if apply_golden_rule:
            self.apply_golden_rule(input_ids, output_ids, logits, golden_rule_threshold)

        return "".join(
            letter + diac
            for (letter, diac) in zip(
                self.sequence_to_input(input_ids), self.sequence_to_target(output_ids)
            )
        )
            

    @cached_property
    def target_id_to_label(self):
        ret = {}
        for idx, symbol in self.target_id_to_symbol.items():
            ret[idx] = DIACRITIC_LABELS.get(symbol, symbol)
        return OrderedDict(sorted(ret.items()))

    def dump_tokens(self) -> dict[Any, Any]:
        data = {
            "pad": self.config.pad,
            "sos": self.config.sos,
            "eos": self.config.eos,
            "input_id_map": dict(self.input_symbol_to_id),
            "target_id_map": dict(self.target_symbol_to_id),
        }
        return {"text_encoder": data}

    def apply_golden_rule(self, input_ids, output_ids, logits, threshold):
        """
        Apply a commonly used rule in MSA that states:
        > If you are uncertain about a **case-ending** diacritic; replace it with **Sukoon**.

        If the model is uncertain about a CE diacritic (as represented by threshold), we use the
        **Sukoon*  diacritic, taking care to apply the rule in only valid cases.
        """
        no_diac_id = self.target_symbol_to_id[ArabicDiacritics.NO_DIACRITIC.value]
        sukoon_id = self.target_symbol_to_id[ArabicDiacritics.SUKOON.value]
        # XXX: Just a personal preference. My TTs pronounces it wrongly
        ta_marbota_id = self.input_symbol_to_id["ة"]
        vowels_ids = frozenset(
            # TODO: remove the `ta_marbota_id` line
            {ta_marbota_id}
            | {self.input_symbol_to_id[v] for v in ARABIC_VOWELS}
        )
        delim_ids = frozenset({self.input_symbol_to_id[d] for d in WORD_DELIMITERS})

        for i in range(len(logits) - 1):
            if (
                input_ids[i + 1] in delim_ids
                and (input_ids[i] not in vowels_ids)
                and (output_ids[i] != no_diac_id)
                and (logits[i] <= threshold)
            ):
                output_ids[i] = sukoon_id
