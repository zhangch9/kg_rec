# -*- coding: utf-8 -*-
"""Contains functions for building neural networks."""


from typing import Callable, Iterable, List, Optional, TypeVar, Union

from torch import nn

T = TypeVar("T")


def ensure_list(x: Union[T, Iterable[T]], size: int) -> List[T]:
    if isinstance(x, Iterable):
        x = list(x)
        if len(x) != size:
            raise ValueError(
                f"The size of the input should be {size}, but got {len(x)}."
            )
    else:
        x = [x] * size
    return x


def get_activation(activation: Optional[Union[str, Callable]]) -> Callable:
    if callable(activation):
        return activation
    if activation is None:
        return nn.Identity()

    activation = activation.lower()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    raise ValueError(f"Activation '{activation}' is not supported.")
