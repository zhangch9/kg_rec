# -*- coding: utf-8 -*-
"""A Knowledge-Graph-Enhanced Recommender System"""


from pathlib import Path

__all__ = ["PACKAGE_ROOT", "PACKAGE_NAME", "PACKAGE_DESCRIPTION"]

PACKAGE_ROOT = Path(__file__).expanduser().resolve().parent
PACKAGE_NAME = PACKAGE_ROOT.name
PACKAGE_DESCRIPTION = "A Knowledge-Graph-Enhanced Recommender System"
