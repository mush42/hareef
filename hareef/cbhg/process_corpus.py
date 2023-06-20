# coding: utf-8

import sys
from .config_manager import ConfigManager
from ..process_corpus import process_corpus_arg_parser, main as proc_main


def main():
    parser = process_corpus_arg_parser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config to be used"
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)
    args.output_dir = config.data_dir
    args.max_chars = config.config["max_len"]

    proc_main(args)


if __name__ == '__main__':
    main()