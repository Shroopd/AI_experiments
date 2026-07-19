from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn import functional as ff

import functional as ff


class AttentionFractal(Module):
    def __init__(
        self,
        dims,
        rank,
    ) -> None:
        super().__init__()
