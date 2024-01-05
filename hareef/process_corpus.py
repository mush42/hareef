# coding: utf-8

import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import reduce, partial
from pathlib import Path

from diacritization_evaluation.util import extract_haraqat
from more_itertools import collapse, windowed
from tqdm import tqdm

from .text_cleaners import collapse_whitespace

_LOGGER = logging.getLogger("hareef.process_corpus")

# Order is critical
SENTENCE_BOUNDRY_PUNCS = [".", "؟", "!", "،", "؛"]
INVALID_SEQUENCES = {
    "َّ": "َّ",
    "ِّ": "ِّ",
    "ُّ": "ُّ",
    "ًّ": "ًّ",
    "ٍّ": "ٍّ",
    "ٌّ": "ٌّ",
    " ،": "،",
    " .": ".",
}


def validate_diacritics(line):
    try:
        text, inputs, diacritics = extract_haraqat(line)
        if "ّ" in diacritics:
            return
        if any(diacritics):
            return text
    except ValueError:
        return


def segment_sentences(max_chars, line):
    return [line.strip() for line in _do_segment_sentences(line, max_chars)]


def add_and_filter_punc(punc, sents, item):
    if item == punc and sents[-1]:
        sents[-1] += punc
    else:
        sents.append(item)
    return sents


def _do_segment_sentences(line, max_chars):
    lines = [
        line,
    ]
    for punc in SENTENCE_BOUNDRY_PUNCS:
        sents = reduce(
            partial(add_and_filter_punc, punc),
            (sent for sent in line.partition(punc) for line in lines),
            [""]
        )
        sents = filter(None, sents)
        lines.clear()
        for sent in sents:
            if 0 < len(sent) <= max_chars:
                sent = collapse_whitespace(sent.rstrip())
                yield sent
            else:
                lines.append(sent)


def normalize_text(text: str):
    for invalid, correct in INVALID_SEQUENCES.items():
        text = text.replace(invalid, correct)
    return text


def take_sample(lines, n) -> (list, list):
    sample = random.sample(lines, n)
    return (list(set(lines).difference(sample)), list(sample))


def write_lines(filename, lines):
    Path(filename).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def process_corpus_arg_parser():
    parser = argparse.ArgumentParser(
        prog="hareef.cbhg.dataset",
        description="Make training, validation, and test datasets from given corpus.",
    )
    parser.add_argument(
        "corpus",
        action="append",
        type=str,
        help="The corpus text file containing lines of diacritized Arabic text",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data", help="Output directory."
    )
    parser.add_argument(
        "--max-chars", type=int, default=400, help="max number of chars per sentence"
    )
    parser.add_argument(
        "--reset-dir",
        action="store_true",
        help="deletes everything under the output directory.",
    )
    parser.add_argument(
        "--n-lines",
        type=int,
        help="apply transformations to a subset of the corpus. Useful for development",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize corpus (strongly recommended)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate lines an remove lines with invalid diacritics",
    )
    parser.add_argument("--n-val", type=int, help="Number of validation sentences")
    parser.add_argument("--n-test", type=int, help="Number of test sentences")
    parser.add_argument(
        "-w",
        "--windowed",
        action="store_true",
        help="Create windowed dataset"
    )
    parser.add_argument(
        "--w-len",
        type=int,
        default=12,
        help="window length in words"
    )
    parser.add_argument(
        "--w-stride",
        type=int,
        default=4,
        help="Steps between adjacent widnows"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of processes used"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="The size of batches sent to each process",
    )
    return parser


def main(args, line_process_func=None):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.n_lines or (args.n_lines > 25000):
        max_workers, chunksize = (args.workers or 8, args.chunk_size or 720)
    else:
        max_workers, chunksize = (
            args.workers or (os.cpu_count() * 4),
            args.chunk_size or round(16e3),
        )

    if args.reset_dir:
        _LOGGER.info("Cleaning output directory...")
        for file in (f for f in output_dir.iterdir() if f.is_file()):
            file.unlink()

    _LOGGER.info("Reading text from corpus...")
    corp_paths = set(Path(c).resolve() for c in args.corpus)
    _LOGGER.info("\n".join(os.fspath(p) for p in corp_paths))
    text = "\n".join(corp.read_text(encoding="utf-8").strip() for corp in corp_paths)
    if args.normalize:
        _LOGGER.info("Normalizing corpus...")
        text = normalize_text(text)

    lines = text.splitlines()

    if args.n_lines:
        _LOGGER.info(f"Sampling maximom of {args.n_lines} lines from the corpus")
        random.shuffle(lines)
        lines = lines[: args.n_lines]

    _LOGGER.info("Removing spurious dots at the beginning of lines...")
    lines = [l.lstrip(".") for l in lines]

    _LOGGER.info("Splitting lines into sentences")
    max_chars = args.max_chars
    _LOGGER.info(f"Maximom length allowed: {max_chars}")
    valid_lines = set(l for l in lines if len(l) <= max_chars)
    invalid_lines = set(lines).difference(valid_lines)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(
            partial(segment_sentences, max_chars), invalid_lines, chunksize=chunksize
        )
        sents = [s for s in tqdm(iterator, total=len(invalid_lines))]
    lines = [*valid_lines, *collapse(sents)]
    _LOGGER.info(f"Num sentences: {len(lines)}")

    if args.validate:
        _LOGGER.info("Removing sentences with invalid or no diacritics...")
        total_lines = len(lines)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(
                validate_diacritics, lines, chunksize=chunksize
            )
            processed_lines = [pl for pl in tqdm(iterator, total=total_lines)]
        lines = list(filter(None, processed_lines))
        _LOGGER.info(f"Ignored: {total_lines - len(lines)}")
        _LOGGER.info(f"Num valid sentences: {len(lines)}")

    if args.windowed:
        _LOGGER.info(
            "Making windowed dataset "
            f"(window length: {args.w_len} words, "
            f"and stride of {args.w_stride}"
        )
        wlen, wstep = args.w_len, args.w_stride
        windows = []
        for line in lines:
            gen_wins = windowed(
                line.split(" "),
                wlen,
                fillvalue="",
                step=wstep
            )
            windows.extend(
                " ".join(win)
                for win in gen_wins
            )
        lines = windows
        _LOGGER.info(f"Total number of generated windows: {len(lines)}")


    _LOGGER.info("Shuffling lines")
    random.shuffle(lines)

    if line_process_func is not None:
        lines = [line_process_func(line) for line in lines]

    n_lines = len(lines)

    _LOGGER.info("Making validation dataset...")
    n_val = args.n_val or round(n_lines * 0.01)
    lines, val_lines = take_sample(lines, n_val)

    _LOGGER.info("Making testing dataset...")
    n_test = args.n_test or round(n_lines * 0.02)
    lines, test_lines = take_sample(lines, n_test)

    _LOGGER.info("Writing lines to text files...")
    write_lines(output_dir.joinpath("train.txt"), lines)
    write_lines(output_dir.joinpath("val.txt"), val_lines)
    write_lines(output_dir.joinpath("test.txt"), test_lines)


if __name__ == "__main__":
    args = process_corpus_arg_parser().parse_args()
    main(args)
