from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn, Tensor
from torch.nn import Module, Linear
from torch.nn import functional as ff

import functional as xff


class Attention(Module):
    def __init__(
        self,
        dims: int,
        **kwargs,  # dtype, device
    ) -> None:
        super().__init__()

        def lin() -> Linear:
            return Linear(dims, dims, False, **kwargs)

        self.key_proj: Linear = lin()
        self.query_proj: Linear = lin()
        self.logit_proj: Linear = lin()
        self.value_proj: Linear = lin()
        self.output_proj: Linear = lin()
