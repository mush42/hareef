# coding: utf-8

import os
import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from more_itertools import collapse
from diacritization_evaluation.util import extract_haraqat

from config_manager import ConfigManager

# Order is critical
SENTENCE_BOUNDRY_PUNCS = [".", "،", "؛", ":"]

INVALID_HARAKA_REPLACE = {
    "َّ": "َّ",
    "ِّ": "ِّ",
    "ُّ": "ُّ",
    "ًّ": "ًّ",
    "ٍّ": "ٍّ",
    "ٌّ": "ٌّ",
    "ّْ": "ّْ",
}


def validate_diacritics(line):
    try:
        text, inputs, diacritics = extract_haraqat(line)
        if any(diacritics):
            return text
    except ValueError:
        return


def segment_sentences(max_chars, line):
    return list(_do_segment_sentences(line, max_chars))


def _do_segment_sentences(line, max_chars):
    lines = [line,]
    for punc in SENTENCE_BOUNDRY_PUNCS:
        sents = [sent for sent in line.split(punc) for line in lines]
        lines.clear()
        for sent in sents:
            if 0 < len(sent) <= max_chars:
                yield sent.rstrip()
            else:
                lines.append(sent)


def fix_invalid_haraka(text: str):
    for invalid, correct in INVALID_HARAKA_REPLACE.items():
        text = text.replace(invalid, correct)
    return text


def take_sample(lines, n) -> (list, list):
    sample = random.sample(lines, n)
    return (list(set(lines).difference(sample)), list(sample))


def write_lines(filename, lines):
    Path(filename).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description ="Make training, validation, and test datasets from given corpus.")
    parser.add_argument(
        "--corpus",
        action="append",
        type=str,
        required=True,
        help="The corpus text file containing lines of diacritized Arabic text",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--reset-dir",
        action="store_true",
        help="deletes everything under this config's folder.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="apply transformations to a subset of the corpus for testing",
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
        "--batch-size",
        type=int,
        default=0,
        help="The size of batches sent to each process",
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)

    output_dir = Path(config.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        max_workers, chunksize = (args.workers or 8, args.batch_size or 720)
    else:
        max_workers, chunksize = (args.workers or (os.cpu_count() * 4), args.batch_size or round(16e3))

    if args.reset_dir:
        print("Cleaning output directory...")
        for file in (f for f in output_dir.iterdir() if f.is_file()):
            file.unlink()

    print("Reading text from corpus...")
    corp_paths = set(Path(c).resolve() for c in args.corpus)
    print("\n".join(os.fspath(p) for p in corp_paths))
    text = "\n".join(
        corp.read_text(encoding="utf-8").strip()
        for corp in corp_paths
    )
    print("Fixing invalid haraka chars in corpus...")
    text = fix_invalid_haraka(text)

    lines = text.splitlines()

    if args.debug:
        print("debug is on. Sampling 5000 lines from the corpus")
        random.shuffle(lines)
        lines = lines[:5000]

    print("Removing spurious dots at the beginning of lines...")
    lines = [l.lstrip(".") for l in lines]

    print("Splitting lines into sentences")
    max_chars = config.config["max_len"]
    print(f"Maximom length allowed: {max_chars}")
    valid_lines = set(l for l in lines if len(l) <= max_chars)
    invalid_lines = set(lines).difference(valid_lines)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sents = executor.map(
            partial(segment_sentences, max_chars),
            invalid_lines,
            chunksize=chunksize
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
    write_lines(output_dir.joinpath("eval.txt"), val_lines)
    write_lines(output_dir.joinpath("test.txt"), test_lines)


if __name__ == "__main__":
    main()
