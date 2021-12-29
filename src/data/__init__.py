# -*- coding: utf-8 -*-
"""API for input pipelines."""


from .datasets import RippleDataset
from .utils import create_dataloader

__all__ = ["RippleDataset", "create_dataloader"]
