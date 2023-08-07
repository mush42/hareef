# coding: utf-8

import argparse
import logging
import os
import string
import subprocess
import sys
import typing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

from tqdm import tqdm


SENT_SEG = [
    "sbd",
    "wtd",
    "none",
]
_LOGGER = logging.getLogger("doc2sents")


def import_string(import_name: str, silent: bool = False) -> typing.Any:
    """
    Imports an object based on a string.
    An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If `silent` is True the return value will be `None` if the import fails.

    :param import_name: the dotted name for the object to import.
    :param silent: if set to `True` import errors are ignored and
                   `None` is returned instead.
    :return: imported object
    """
    import_name = import_name.replace(":", ".")
    try:
        try:
            __import__(import_name)
        except ImportError:
            if "." not in import_name:
                raise
        else:
            return sys.modules[import_name]

        module_name, obj_name = import_name.rsplit(".", 1)
        module = __import__(module_name, globals(), locals(), [obj_name])
        try:
            return getattr(module, obj_name)
        except AttributeError as e:
            raise ImportError(e) from None

    except ImportError as e:
        if not silent:
            raise ImportStringError(import_name, e).with_traceback(
                sys.exc_info()[2]
            ) from None

    return None


def format_as_table(*cols: tuple[str, list[str]]) -> str:
    """Taken from lightening"""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width
    return summary


def get_stats(sentences):
    num_words = [len(sent.split(" ")) for sent in sentences]
    num_chars = [len(sent) for sent in sentences]
    data = [num_chars, num_words]
    cols = (
        ("", ["  chars", "  words"]),
        ("mean", [round(mean(d)) for d in data]),
        ("median", [median(d) for d in data]),
        ("min", [min(d) for d in data]),
        ("max", [max(d) for d in data]),
    )
    return format_as_table(*cols)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="doc2sents", description="Convert a document to a list of sentences"
    )
    parser.add_argument("doc", type=Path, help="Source document")
    parser.add_argument(
        "-o", "--output", type=Path, required=False, help="Output plain text file"
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        required=False,
        help="source document language for sentence segmentation",
    )
    parser.add_argument(
        "-c",
        "--cleaner",
        type=str,
        required=False,
        help="dotted path to a python callable to clean input lines",
    )
    parser.add_argument(
        "-s",
        "--seg",
        type=str,
        choices=SENT_SEG,
        default="none",
        help="package used for segmenting input text to sentences",
    )
    parser.add_argument(
        "--wtp-model", type=str, required=False, help="WTP model name or path"
    )
    parser.add_argument(
        "--drop-single-word", action="store_true", required=False, help="drop single word sentences"
    )
    parser.add_argument(
        "--drop-punc", action="store_true", required=False, help="drop punctuation only lines"
    )
    parser.add_argument(
        "--stats", action="store_true", required=False, help="show statistics about character and word counts"
    )


    args = parser.parse_args()

    if not os.path.exists(args.doc):
        _LOGGER.error(f"File not found: ` {args.doc}`")
        sys.exit(-1)

    _LOGGER.info(f"Using document: {args.doc}")

    cleaner_func = lambda s: s
    if args.cleaner:
        _LOGGER.info(f"importing cleaner callable: {args.cleaner}")
        try:
            cleaner_func = import_string(args.cleaner)
        except:
            _LOGGER.error(
                f"Failed to import cleaner function: {args.cleaner}", exc_info=True
            )
            sys.exit(-1)

    _LOGGER.info("converting document to plain text using pandoc")
    try:
        stdout = subprocess.check_output(
            ["pandoc", "--to", "plain", "--wrap", "none", args.doc]
        )
        plain_text = stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        _LOGGER.error(
            f"Failed to convert document to plain text. Pandoc exited with a non-zero exit code: {e.returncode}"
        )
        _LOGGER.error(e.output)
        sys.exit(-1)

    _LOGGER.info("cleaning and stripping blank lines")
    lines = [line.strip() for line in plain_text.splitlines()]
    lines = {cleaner_func(line): None for line in lines}
    lines.pop("", None)
    lines = list(lines)

    _LOGGER.info("segmenting lines to sentences")
    sentences = lines
    if args.seg == "sbd":
        _LOGGER.info("using `pysbd` segmenter")
        from pysbd import Segmenter

        sentences = []
        sent_segmenter = Segmenter(language=args.lang)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            iterator = executor.map(
                sent_segmenter.segment, lines, chunksize=len(lines) // os.cpu_count()
            )
            for sents in tqdm(iterator, total=len(lines)):
                sentences.extend(sents)
    elif args.seg == "wtp":
        _LOGGER.info("using `wtp` segmenter")
        if not args.wtp_model:
            _LOGGER.error("`--wtp-model is required if using WTP segmenter")
            sys.exit(-1)
        from wtpsplit import WtP

        wtp = WtP(args.wtp_model)
        sentences = []
        for line in tqdm(lines):
            sentences.extend(wtp.split(line, lang_code=args.lang))

    sentences = list({sent.strip(): None for sent in sentences})

    if args.drop_single_word:
        sentences = [
            sent
            for sent in sentences
            if len(sent.split(" ")) > 1
        ]

    if args.drop_punc:
        punctuation = set(string.punctuation)
        sentences = [
            sent
            for sent in sentences
            if set(sent).difference(punctuation)
        ]

    sentences = {sent.strip(): None for sent in sentences}
    sentences.pop("", None)
    sentences = list(sentences)

    if not args.output:
        doc_path = Path(args.doc)
        args.output = os.fspath(doc_path.parent.joinpath(f"{doc_path.stem}.txt"))

    _LOGGER.info(f"Writing sentences to output file: {args.output}")

    Path(args.output).write_text("\n".join(sentences), encoding="utf-8", newline="\n")

    _LOGGER.info(f"Wrote {len(sentences)} sentences to file `{args.output}`")
    if args.stats:
        stats = get_stats(sentences)
        _LOGGER.info(f"Stats:\n{stats}")


if __name__ == "__main__":
    main()
