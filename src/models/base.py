# -*- coding: utf-8 -*-
"""Contains the BaseModel class, from which all models inherit."""


from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
from jsonargparse import Namespace
from jsonargparse.util import import_object


class BaseModel(pl.LightningModule):

    _ignore_decay: Tuple[str, ...] = ("bias",)

    def __init__(self, optim_args: Namespace):
        """Base class for all models.

        Args:
            optim_args: Arguments for optimization.
        """
        super().__init__()
        self._optim_args = optim_args

    def _should_decay(self, name: str) -> bool:
        for pattern in self._ignore_decay:
            if pattern in name:
                return False
        return True

    def _get_param_groups(self) -> List[Dict[str, Any]]:
        weight_decay = getattr(self._optim_args, "weight_decay", 0.0)
        enable_decay = []
        disable_decay = []
        memory = set()
        modules = self.modules()
        for m in modules:
            for name, param in m._parameters.items():
                if param is None or param in memory:
                    continue
                memory.add(param)
                if not param.requires_grad:
                    continue
                if self._should_decay(name):
                    enable_decay.append(param)
                else:
                    disable_decay.append(param)
        return [
            {"params": enable_decay, "weight_decay": weight_decay},
            {"params": disable_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        param_groups = self._get_param_groups()
        optimizer = import_object(self._optim_args.optimizer.class_path)(
            param_groups, **self._optim_args.optimizer.init_args
        )
        return optimizer
