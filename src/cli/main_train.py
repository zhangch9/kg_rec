# -*- coding: utf-8 -*-
"""Entrypoint for command `train'."""


import copy
import logging
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse.actions import _ActionSubCommands
from jsonargparse.util import import_object
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim import Optimizer

from ..models import BaseModel
from .logging_utils import add_options_logging, initialize_logging

import dgl  # isort: skip


logger = logging.getLogger(__name__)


def generate_parser(
    sub_commands: Optional[_ActionSubCommands] = None,
) -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a neural network for item recommendation."
    )
    if sub_commands:
        sub_commands.add_subcommand("train", parser, help=parser.description)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Train the model for this many times.",
    )
    parser.add_argument(
        "--seed",
        type=Optional[int],
        help=(
            "If set to an integer, will use this value "
            "to seed the random state."
        ),
    )
    # Augments for the model.
    parser.add_subclass_arguments(
        BaseModel,
        "model",
        as_group=True,
        instantiate=False,
        required=True,
        skip={"optim_args"},
    )
    # Arguments for optimization.
    _add_options_optim(parser)
    # Augments for the trainer.
    parser.add_class_arguments(
        pl.Trainer,
        "trainer",
        as_group=True,
        as_positional=True,
        instantiate=True,
        skip={"callbacks", "logger", "deterministic"},
    )
    parser.add_subclass_arguments(
        LightningLoggerBase,
        "trainer.logger",
        as_group=True,
        instantiate=False,
        required=False,
    )
    parser.set_defaults(
        {
            "trainer.num_sanity_val_steps": -1,
            "trainer.detect_anomaly": True,
        }
    )
    _add_options_early_stopping(parser)
    _add_options_checkpoint(parser)
    add_options_logging(parser)
    return parser


def _add_options_optim(parser: ArgumentParser):
    parser.add_subclass_arguments(
        Optimizer,
        "optim.optimizer",
        as_group=True,
        instantiate=True,
        required=True,
        skip={"params"},
    )
    options_optim = parser.add_argument_group("Options for Optimization")
    options_optim.add_argument(
        "--optim.grad_clip_val",
        type=float,
        default=0.0,
        help="A value at which to clip gradients.",
    )


def _add_options_early_stopping(parser: ArgumentParser):
    options_es = parser.add_argument_group("Options for Early Stopping")
    options_es.add_argument(
        "--early_stopping.monitor",
        type=Optional[str],
        default=None,
        help=(
            "Metric to be monitored. If set to `None`, will disable "
            "early-stopping"
        ),
    )
    options_es.add_argument(
        "--early_stopping.patience",
        type=int,
        default=0,
        help=(
            "Number of checks with no improvement after which training "
            "will be stopped. If set to 0, will disable early-stopping."
        ),
    )
    options_es.add_argument(
        "--early_stopping.mode",
        choices=("min", "max"),
        default="max",
        help=(
            "If set to `min`, training will stop when the metric monitored has "
            "stopped stopped decreasing; If set to `max`, training will stop "
            "when the metric monitored has stopped increasing."
        ),
    )


def _add_options_checkpoint(parser: ArgumentParser):
    options_ckpt = parser.add_argument_group("Options for Model Checkpoints")
    options_ckpt.add_argument(
        "--checkpoint.monitor",
        type=Optional[str],
        default=None,
        help=(
            "Metric to be monitored. The value is overridden by "
            "'--early_stopping.monitor'"
        ),
    )
    options_ckpt.add_argument(
        "--checkpoint.save_top_k",
        type=int,
        default=0,
        help=(
            "If set to a positive integer, the best k models according to "
            "the metric monitored will be saved."
        ),
    )
    options_ckpt.add_argument(
        "--checkpoint.mode",
        choices=("min", "max"),
        default=None,
        help=(
            "If set to `min`, training will stop when the metric monitored has "
            "stopped stopped decreasing; If set to `max`, training will stop "
            "when the metric monitored has stopped increasing. The value is "
            "overridden by '--early_stopping.mode'."
        ),
    )
    options_ckpt.add_argument(
        "--checkpoint.every_n_epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints.",
    )


def train(args: Namespace) -> Dict[str, Union[int, float]]:
    if args.seed is not None:
        pl.seed_everything(args.seed)
        dgl.seed(args.seed)
        args.trainer.deterministic = True

    # We instantiate the model.
    model = import_object(args.model.class_path)(
        optim_args=args.optim, **args.model.init_args
    )
    logger.info(f"model architecture:\n{model}")

    # We instantiate the trainer.
    args_logger = getattr(args.trainer, "logger", None)
    if args_logger is None:
        args.trainer.logger = False
    else:
        args.trainer.logger = import_object(args_logger.class_path)(
            **args_logger.init_args
        )
    args.trainer.callbacks = []
    if (
        args.early_stopping.monitor is not None
        and args.early_stopping.patience > 0
    ):
        args.trainer.callbacks.append(
            pl.callbacks.EarlyStopping(
                strict=True,
                check_finite=True,
                check_on_train_epoch_end=False,
                **args.early_stopping,
            )
        )
    args.checkpoint.monitor = (
        args.early_stopping.monitor or args.checkpoint.monitor
    )
    args.checkpoint.mode = args.early_stopping.mode or args.checkpoint.mode
    args.trainer.callbacks.append(
        pl.callbacks.ModelCheckpoint(save_last=True, **args.checkpoint)
    )
    trainer = pl.Trainer(**args.trainer)

    trainer.fit(model)
    if args.checkpoint.monitor is not None and args.checkpoint.save_top_k > 0:
        metrics = trainer.test(ckpt_path="best")
    else:
        metrics = trainer.test(model, ckpt_path=None)
    return metrics[0]


def main(args: Namespace):
    initialize_logging(args.pop("verbose"))
    num_runs = args.pop("num_runs")
    if num_runs > 1:
        args.seed = None
    metrics_per_run = []
    for _ in range(num_runs):
        metrics_per_run.append(train(copy.deepcopy(args)))
    metrics = metrics_per_run[0]
    print(f"Summary of {len(metrics_per_run)} runs:")
    for k in metrics:
        values = np.asarray([m[k] for m in metrics_per_run])
        print(f"{k}_avg = {np.mean(values)}, {k}_std = {np.std(values)}")
