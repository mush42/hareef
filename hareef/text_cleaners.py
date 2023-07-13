# coding: utf-8

import re

from .constants import DIACRITIC_CHARS, VALID_ARABIC_CHARS

_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    text = re.sub(_whitespace_re, " ", text)
    return text


def basic_cleaner(text):
    text = collapse_whitespace(text)
    return text.strip()


def valid_arabic_cleaner(text):
    text = filter(lambda char: char in VALID_ARABIC_CHARS, text)
    text = collapse_whitespace("".join(list(text)))
    return text.strip()


def diacritics_cleaner(text: str) -> str:
    return text.translate(str.maketrans("", "", "".join(DIACRITIC_CHARS)))
