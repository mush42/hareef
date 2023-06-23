# coding: utf-8

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from diacritization_evaluation.util import extract_haraqat
from more_itertools import collapse

from .text_cleaners import collapse_whitespace


# Order is critical
SENTENCE_BOUNDRY_PUNCS = [".", "؟", "!", "،", "؛"]

INVALID_SEQUENCES = {
    "َّ": "َّ",
    "ِّ": "ِّ",
    "ُّ": "ُّ",
    "ًّ": "ًّ",
    "ٍّ": "ٍّ",
    "ٌّ": "ٌّ",
    "ّْ": "ّْ",
    " .": "."
}


def validate_diacritics(line):
    try:
        text, inputs, diacritics = extract_haraqat(line)
        if any(diacritics):
            return text
    except ValueError:
        return


def segment_sentences(max_chars, line):
    return [line.strip() for line in _do_segment_sentences(line, max_chars)]


def _do_segment_sentences(line, max_chars):
    lines = [
        line,
    ]
    for punc in SENTENCE_BOUNDRY_PUNCS:
        sents = [sent for sent in line.split(punc) for line in lines]
        lines.clear()
        for sent in sents:
            if 0 < len(sent) <= max_chars:
                sent = collapse_whitespace(sent.rstrip())
                # eliminate very short sentences
                if sent.count(' ') < 3:
                    continue
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
        description="Make training, validation, and test datasets from given corpus."
    )
    parser.add_argument(
        "corpus",
        action="append",
        type=str,
        help="The corpus text file containing lines of diacritized Arabic text",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory."
    )
    parser.add_argument("--max-chars", type=int, default=400, help="max number of chars per sentence")
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
        "--validate",
        action="store_true",
        help="Validate lines an remove lines with invalid diacritics",
    )
    parser.add_argument("--n-val", type=int, help="Number of validation sentences")
    parser.add_argument("--n-test", type=int, help="Number of test sentences")
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


def main(args):
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
        print("Cleaning output directory...")
        for file in (f for f in output_dir.iterdir() if f.is_file()):
            file.unlink()

    print("Reading text from corpus...")
    corp_paths = set(Path(c).resolve() for c in args.corpus)
    print("\n".join(os.fspath(p) for p in corp_paths))
    text = "\n".join(corp.read_text(encoding="utf-8").strip() for corp in corp_paths)
    print("Normalizing corpus...")
    text = normalize_text(text)

    lines = text.splitlines()

    if args.n_lines:
        print(f"Sampling maximom of {args.n_lines} lines from the corpus")
        random.shuffle(lines)
        lines = lines[:args.n_lines]

    print("Removing spurious dots at the beginning of lines...")
    lines = [l.lstrip(".") for l in lines]

    print("Splitting lines into sentences")
    max_chars = args.max_chars
    print(f"Maximom length allowed: {max_chars}")
    valid_lines = set(l for l in lines if len(l) <= max_chars)
    invalid_lines = set(lines).difference(valid_lines)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sents = executor.map(
            partial(segment_sentences, max_chars), invalid_lines, chunksize=chunksize
        )
    lines = [*valid_lines, *collapse(sents)]
    print(f"Num sentences: {len(lines)}")

    if args.validate:
        print("Removing sentences with invalid or no diacritics...")
        total_lines = len(lines)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            processed_lines = executor.map(
                validate_diacritics, lines, chunksize=chunksize
            )
        lines = list(filter(None, processed_lines))
        print(f"Ignored: {total_lines - len(lines)}")
        print(f"Num valid sentences: {len(lines)}")

    print("Shuffling lines")
    random.shuffle(lines)

    n_lines = len(lines)

    print("Making validation dataset...")
    n_val = args.n_val or round(n_lines * 0.01)
    lines, val_lines = take_sample(lines, n_val)

    print("Making testing dataset...")
    n_test = args.n_test or round(n_lines * 0.05)
    lines, test_lines = take_sample(lines, n_test)

    print("Writing lines to text files...")
    write_lines(output_dir.joinpath("train.txt"), lines)
    write_lines(output_dir.joinpath("val.txt"), val_lines)
    write_lines(output_dir.joinpath("test.txt"), test_lines)


if __name__ == "__main__":
    args = process_corpus_arg_parser().parse_args()
    main(args)
