# -*- coding: utf-8 -*-
"""Contains functions for logging."""


import logging

from jsonargparse import ArgumentParser

LOG_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def initialize_logging(level: int):
    logging.captureWarnings(True)
    if level > 0:
        logging.basicConfig(level=LOG_LEVEL[min(2, level)])


def add_options_logging(parser: ArgumentParser):
    options_logging = parser.add_argument_group("Logging Options")
    options_logging.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Enable verbose mode. Multiple -v options increase the verbosity "
            "level, i.e., -v for INFO, -vv for DEBUG."
        ),
    )
