# coding: utf-8

"""
Constants that are used by all models
"""

import enum


class ArabicDiacritics(enum.Enum):
    """All possible Arabic diacritics."""

    NO_DIACRITIC = ""
    SUKOON = "ْ"
    SHADDA = "ّ"
    DAMMA = "ُ"
    FATHA = "َ"
    KASRA = "ِ"
    TANWEEN_DAMMA = "ٌ"
    TANWEEN_FATHA = "ً"
    TANWEEN_KASRA = "ٍ"
    SHADDA_PLUS_DAMMA = "ُّ"
    SHADDA_PLUS_FATHA = "َّ"
    SHADDA_PLUS_KASRA = "ِّ"
    SHADDA_PLUS_TANWEEN_DAMMA = "ٌّ"
    SHADDA_PLUS_TANWEEN_FATHA = "ًّ"
    SHADDA_PLUS_TANWEEN_KASRA = "ٍّ"

    @classmethod
    def chars(cls):
        return {
            cls.SUKOON,
            cls.SHADDA,
            cls.DAMMA,
            cls.FATHA,
            cls.KASRA,
            cls.TANWEEN_DAMMA,
            cls.TANWEEN_FATHA,
            cls.TANWEEN_KASRA,
        }

    @classmethod
    def valid(cls):
        return {
            cls.NO_DIACRITIC,
            cls.SUKOON,
            cls.SHADDA,
            cls.DAMMA,
            cls.FATHA,
            cls.KASRA,
            cls.TANWEEN_DAMMA,
            cls.TANWEEN_FATHA,
            cls.TANWEEN_KASRA,
            cls.SHADDA_PLUS_DAMMA,
            cls.SHADDA_PLUS_FATHA,
            cls.SHADDA_PLUS_KASRA,
            cls.SHADDA_PLUS_TANWEEN_DAMMA,
            cls.SHADDA_PLUS_TANWEEN_FATHA,
            cls.SHADDA_PLUS_TANWEEN_KASRA,
        }

    @classmethod
    def diacritic_to_label(cls):
        return {
            member.value: name
            for (name, member) in cls.__members__.items()
        }


WORD_SEPARATOR = chr(0x20)
ARABIC_LETTERS = frozenset(
    {chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))}
)
PUNCTUATIONS = frozenset({".", "،", ":", "؛", "-", "؟", "!", "(", ")", "[", "]", '"', "«", "»",})
DIACRITIC_CHARS = {diac.value for diac in ArabicDiacritics.chars()}
ALL_VALID_DIACRITICS = {m.value for m in ArabicDiacritics.valid()} 
DIACRITIC_LABELS = ArabicDiacritics.diacritic_to_label()
VALID_ARABIC_CHARS = {WORD_SEPARATOR, *ARABIC_LETTERS, *PUNCTUATIONS, *DIACRITIC_CHARS}
ARABIC_VOWELS = {
    chr(c)
    for c in [0x621, 0x622, 0x623, 0x624, 0x625, 0x626, 0x627, 0x648, 0x649, 0x64a]
}
SENTENCE_DELIMITERS = {".", "؟", "!", "،", ":", "؛", "(", ")", "[", "]", '"', "«", "»",}
WORD_DELIMITERS = {WORD_SEPARATOR, *SENTENCE_DELIMITERS}

