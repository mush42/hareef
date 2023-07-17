# coding: utf-8

import logging
import sys

from ..process_corpus import main as proc_main
from ..process_corpus import process_corpus_arg_parser

from .config import Config


def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = process_corpus_arg_parser()
    parser.add_argument(
        "--config", type=str, required=True, help="Model config to be used"
    )
    args = parser.parse_args()

    config = Config(args.config)
    args.output_dir = config.data_dir
    args.max_chars = config.config["max_len"]

    proc_main(args)


if __name__ == "__main__":
    main()
