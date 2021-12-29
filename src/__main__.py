# -*- coding: utf-8 -*-
"""Entrypoint module, in case you use `python -m`."""


import sys

from .cli import main

sys.exit(main(sys.argv[1:]))
