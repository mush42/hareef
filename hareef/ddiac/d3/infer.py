# coding: utf-8

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import torch as T
from pyarabic.araby import tokenize
from tqdm import tqdm

from hareef.utils import find_last_checkpoint
from ..config import Config
from ..dataset import DataRetriever, load_test_data
from .model import DiacritizerD3


_LOGGER = logging.getLogger(__package__)
DIACRITICS = {
    "FATHA": 1,
    "KASRA": 2,
    "DAMMA": 3,
    "SUKUN": 4
}



class D3Predictor: 

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.data_utils = self.config.data_utils
        self.data_loader = load_test_data(self.config, is_test=True)

        self.mapping = self.data_utils.load_mapping_v3('test')
        self.original_lines = self.data_utils.load_file_clean('test', strip=True)

    def shakkel_char(self, diac: int, tanween: bool, shadda: bool) -> str:
        text = ""
        if shadda and diac != DIACRITICS["SUKUN"]:
            text += chr(0x651)

        if diac == DIACRITICS["FATHA"]:
            text += chr(0x64E) if not tanween else chr(0x64B)
        elif diac == DIACRITICS["KASRA"]:
            text += chr(0x650) if not tanween else chr(0x64D)
        elif diac == DIACRITICS["DAMMA"]:
            text += chr(0x64F) if not tanween else chr(0x64C)
        elif diac == DIACRITICS["SUKUN"]:
            text += chr(0x652)

        return text

    def predict(self):
        y_gen_diac, y_gen_tanween, y_gen_shadda = self.model.predict(self.data_loader)        
        diacritized_lines = []
        _LOGGER.info("Processing lines")
        for sent_idx, line in tqdm(enumerate(self.original_lines), total=len(self.original_lines)):
            diacritized_line = ""
            line = ' '.join(tokenize(line))
            for char_idx, char in enumerate(line):
                diacritized_line += char
                char_vote_haraka, char_vote_shadda, char_vote_tanween = [], [], []
                if sent_idx not in self.mapping: continue
                for seg_idx in self.mapping[sent_idx]:
                    for t_idx in self.mapping[sent_idx][seg_idx]: 
                        if char_idx in self.mapping[sent_idx][seg_idx][t_idx]:
                            c_idx = self.mapping[sent_idx][seg_idx][t_idx].index(char_idx)
                            try:
                                char_vote_haraka  += [y_gen_diac[seg_idx][t_idx][c_idx]]
                                char_vote_shadda  += [y_gen_shadda[seg_idx][t_idx][c_idx]]
                                char_vote_tanween += [y_gen_tanween[seg_idx][t_idx][c_idx]]
                            except IndexError:
                                pass

                if len(char_vote_haraka) > 0:
                    char_mv_diac = Counter(char_vote_haraka).most_common()[0][0]
                    char_mv_shadda = Counter(char_vote_shadda).most_common()[0][0]
                    char_mv_tanween = Counter(char_vote_tanween).most_common()[0][0]
                    diacritized_line += self.shakkel_char(char_mv_diac, char_mv_tanween, char_mv_shadda)
            
            diacritized_lines += [diacritized_line.strip()]
        return diacritized_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hareef.ddiac.d2.infer')
    parser.add_argument('--config', type=str, required=True, help='path of config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint to use for inference')
    args = parser.parse_args()

    config = Config(args.config)

    if not args.checkpoint:
        try:
            checkpoint_filename, epoch, step = find_last_checkpoint(
                config["paths"]["logs"]
            )
            _LOGGER.info(f"Using checkpoint from: epoch={epoch} - step={step}")
            _LOGGER.info(f"file: {checkpoint_filename}")
            args.checkpoint = checkpoint_filename
        except:
            _LOGGER.exception(
                "Failed to obtain the path to the last checkpoint", exc_info=True
            )
            sys.exit()

    model = DiacritizerD3(config)

    predictor = D3Predictor(model)
    diacritized_lines = predictor.predict()

    exp_id = config["session"].split("-")[-1].lower()

    target_file = Path(config["paths"]["logs"]).joinpath('predictions', f'predictions_{exp_id}.txt')
    target_file.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Writing predictions to file: `{target_file}`")
    with open(target_file, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(diacritized_lines))
