#! -*- coding: utf-8 -*-
"""Builds the command line interface."""


from jsonargparse import ArgumentParser

from .. import PACKAGE_DESCRIPTION, PACKAGE_NAME
from . import main_create, main_train


def generate_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog=PACKAGE_NAME,
        description=PACKAGE_DESCRIPTION,
    )

    sub_commands = parser.add_subcommands(
        dest="command", title="Available Commands", metavar="COMMAND"
    )

    main_create.generate_parser(sub_commands)
    main_train.generate_parser(sub_commands)

    return parser
