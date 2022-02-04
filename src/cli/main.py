# -*- coding: utf-8 -*-
"""Main entry point for CLI."""


import time
from typing import Sequence, Text

from . import main_create, main_train
from .argparse import generate_parser

COMMAND_MODULE = {"create": main_create, "train": main_train}


def main(argv: Sequence[Text]):
    start = time.perf_counter()
    parser = generate_parser()
    if len(argv) == 0:
        argv = ["-h"]
    args = parser.parse_args(argv)
    # execute subcommand
    fn = getattr(COMMAND_MODULE[args.command], "main")
    args = getattr(args, args.command)
    fn(args)
    print(f"Running time: {time.perf_counter() - start:.3f}s.")
