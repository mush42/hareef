# coding: utf-8

import argparse
import logging
import os
from pathlib import Path

from hareef.process_corpus import main as proc_main, process_corpus_arg_parser, validate_diacritics

import fasttext as ft
from tqdm import tqdm
from pyarabic.araby import tokenize, strip_tashkeel

from .config import Config

_LOGGER = logging.getLogger("hareef.process_corpus")


def export(path, text):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(text))
    
def segment(lines, stride, window_sz, min_window_sz):
    segments, mapping = [], []
    real_seg_idx = 0

    for sent_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        tokens = tokenize(line)
        if len(tokens) == 0: continue
        if tokens[-1] == '\n': tokens = tokens[:-1]
        seg_idx, idx = 0, 0
        while idx < len(tokens):
            window = tokens[idx:idx+window_sz]
            if window_sz == -1: window = tokens  
            if len(window) < min_window_sz and seg_idx != 0: break

            segment = ' '.join(window)
            segments += [segment]
            char_offset = len(strip_tashkeel(' '.join(tokens[:idx])))
    
            if seg_idx > 0:
                char_offset += 1

            seg_tokens = tokenize(strip_tashkeel(segment))

            j = 0
            for st_idx, st in enumerate(seg_tokens):
                for _ in range(len(st)):
                    mapping += [(sent_idx, real_seg_idx, st_idx, j+char_offset)]
                    j += 1
                j += 1

            real_seg_idx += 1
            seg_idx += 1

            if stride == -1: break

            idx += (window_sz if stride >= window_sz else stride)
          
    return segments, mapping


def extract_vocab(f_name):
    vocab = set()
    with open(f_name, 'r', encoding="utf-8") as fin:
        lines = fin.readlines()
    for line in tqdm(lines):
        vocab.update(
            strip_tashkeel(t)
            for t in tokenize(line)
        )
    return vocab


def read_vocab(fn):
    with open(fn, encoding='utf-8') as fin:
        vv = [w.strip() for w in tqdm(fin) if w.strip()]
    return vv

def embed_vocab(ff, vv):
    return [
        ff.get_word_vector(w)
        for w in tqdm(vv)
    ]

def vround(val):
    return str(round(val, 4))

def render(word, feats):
    ff = ' '.join(map(vround, feats))
    return f'{word} {ff}\n'


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = process_corpus_arg_parser()
    parser.add_argument('--config',  type=str, required=True, help='Model config')
    parser.add_argument('--segment',  type=str, choices=['train', 'test'], default='train', help='Model config')
    parser.add_argument('--ft-model',  type=str, required=True, help='Fast text binary')

    args = parser.parse_args()

    config = Config(args.config)

    base = config["paths"]["base"]
    _LOGGER.info(f"Base dataset dir: {base}")

    args.output_dir = os.path.join(base, "dataset")
    args.max_chars = config["train"]["max-word-len"] * config["train"]["max-sent-len"]
    _LOGGER.info(f"Max chars allowed in sentence: {args.max_chars}")
    proc_main(args)

    # Vocab
    _LOGGER.info("Extracting vocab from corpus...")
    v_train = extract_vocab(os.path.join(base, "dataset", "train.txt"))
    v_val = extract_vocab(os.path.join(base, "dataset", "val.txt"))
    v_test = extract_vocab(os.path.join(base, "dataset", "test.txt"))
    vocab = v_train.union(v_val).union(v_test)
    vocab_filepath = os.path.join(base, "vocab.txt")
    with open(vocab_filepath, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(vocab))
    _LOGGER.info(f"Wrote vocab to file: {vocab_filepath}")

    # Word embedding
    _LOGGER.info("Extracting word embedding...")
    _LOGGER.info(f"Using vocab list from file: {vocab_filepath}")
    words = sorted(read_vocab(vocab_filepath))
    _LOGGER.info(f"Vocab size: {len(words)}")
    ember = ft.load_model(args.ft_model)
    ember_length = ember.get_dimension()
    vocab = embed_vocab(ember, words)
    del ember

    save_path = config["paths"]["word-embs"]
    with open(save_path, 'w', encoding="utf-8") as fout:
        fout.write(f'{len(vocab)} {ember_length}\n')
        fout.writelines(
            render(word, feats)
            for word, feats in zip(tqdm(words), vocab)
        )
    _LOGGER.info(f"Saved word embedding to file: {save_path}")

    # Segmentation
    _LOGGER.info("Segmenting corpus using sliding window...")
    if args.segment == 'train':
        seg_config = config["segment"]["train"]
        _LOGGER.info("Using segment config for training")
    else:
        seg_config = config["segment"]["test"]
        _LOGGER.info("Using segment config for test")

    stride = seg_config["stride"]
    window = seg_config["window"]
    min_window = seg_config["min-window"]
    export_map = seg_config["export-map"]

    for fpath in tqdm(seg_config["files"]):
        file_path = os.path.join(base, "dataset", fpath)
        _LOGGER.info(f"Processing file: {file_path}")
        save_path = os.path.join(base, fpath[:-4] + f"-{stride}-{window}.txt")
        map_path  = os.path.join(base, fpath[:-4] + f"-{stride}-{window}.map")

        with open(file_path, 'r', encoding="utf-8") as fin:
            lines = fin.readlines()

        segments, mapping = segment(lines, stride, window, min_window)
        segments = list(filter(
            None,
            (validate_diacritics(seg) for seg in segments)
        ))

        with open(save_path, 'w', encoding="utf-8") as fout:
            fout.write('\n'.join(segments))
        _LOGGER.info(f"Saved to file: {save_path}")

        if export_map:
            with open(map_path, 'w', encoding="utf-8") as fout:
                for sent_idx, seg_idx, word_idx, char_idx in mapping:
                    fout.write(f"{sent_idx}, {seg_idx}, {word_idx}, {char_idx}\n")
            _LOGGER.info(f"Exporting map to file: {map_path}")

    _LOGGER.info("Process is complete")


if __name__ == '__main__':
    main()
