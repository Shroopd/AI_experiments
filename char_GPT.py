import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as ff

import my_experiments as mxp


def mask(x: Tensor) -> Tensor:
    return x.triu(diagonal=1)


class MLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.all = nn.Sequential(
            mxp.LinearActivateZP(dims, dims * 2, nn.SiLU()),
            nn.Linear(2 * dims, dims, False),
        )

    def forward(self, X):
        return self.all(X)


class GPT(nn.Module):
    def __init__(self, dims: int, layers: int) -> None:
        super().__init__()
        self.all = nn.Sequential(
            *(mxp.FractalTransformer(dims, 2, MLP, mask=mask) for _ in range(layers))
        )

    def forward(self, X):
        return self.all(X)
