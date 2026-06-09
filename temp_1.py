from __future__ import annotations

import math
import itertools
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as ff
from torch.nn.parameter import Parameter
from torch import Tensor

import general

def compare(
    query: Tensor,
    key: Tensor,
    dim: torch.types._size | int = -2,
) -> Tensor:
    pass


def select(
    value: Tensor,
    scores: Tensor,
    dim: torch.types._size | int = -2,
    attended_input_dim: torch.types._size | int = -2,
    attended_output_dim: torch.types._size | int = -2,
    *,
    # selection_function = torch.softmax,
    selection_function = general.swishmax,
) -> Tensor:
    pass
