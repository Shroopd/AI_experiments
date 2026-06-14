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


def insert_ones(shape: Iterable[int], *dims: int):
    "Not in-place."
    shape = list(shape)
    lenp1 = len(shape) + 1
    target_dims = sorted(d % lenp1 for d in dims)
    target_dims = list(target_dims[i] + i for i in range(len(target_dims)))
    for d in target_dims:
        shape.insert(d, 1)
    return shape, tuple(target_dims)


def unsqueeze_multi(
    X: Tensor,
    *dims: int,
) -> tuple[Tensor, tuple[int, ...]]:
    r"""
    unsqueeze_multi(input, dim) -> Tensor, tuple[int, ...]

    Applies :func:`torch.unsqueeze` to all dimensions in parallel.

    The returned tensor shares the same underlying data with this tensor.

    Also returns a tuple carrying the indices of all new dimensions.

    Args:
        input (Tensor): the input tensor.
        dim (int): the indices at which to insert the singleton dimensions

    Example::

        >>> x = torch.tensor([1, 2, 3, 4])
        >>> unsqueeze_multi(x, 0)[0].shape
        torch.Size([1, 4])
        >>> unsqueeze_multi(x, 1)[0].shape
        torch.Size([4, 1])
        >>> unsqueeze_multi(x, 0, 1)[0].shape
        torch.Size([1, 4, 1])
    """

    # dim_sort = sorted((d % (X.dim() + 1) - X.dim() - 1 for d in dims), reverse=False)
    # for d in dim_sort:
    #     X = X.unsqueeze(d)
    shape, new_dims = insert_ones(X.shape, *dims)
    return X.view(shape), new_dims


def compare(
    lhs: Tensor,  # [..., K, ..., V]
    rhs: Tensor,  # [..., Q, ..., V]
    *dims: int,
    operator: Callable[[Tensor, Tensor], Tensor] = torch.mul,
) -> Tensor:
    """
    compare(lhs,rhs) -> Tensor

    

    Default of torch.mul is basically a matrix multiplication, but the vector dimension is not summed up.
    [a,v] @ [v,b] = [a,b]
    compare([a,v],[b,v]) = [a,b,v]
    
    `A @ B == compare(A,B,-2).sum(-1)`

    Currently cannot compare dimension -1?
    """

    lhs, lhs_dims = unsqueeze_multi(lhs, *(d + 1 for d in dims))
    # [..., K, 1, ..., V]

    rhs, rhs_dims = unsqueeze_multi(rhs, *(d for d in dims))
    # [..., 1, Q, ..., V]

    return operator(lhs, rhs)
    # [..., K, Q, ..., V] = [..., K, 1, ..., V] operator [..., 1, Q, ..., V]


def apply(
    value: Tensor,
    scores: Tensor,
    *dims: int,
) -> Tensor:
    """
    Functional version of the "distribute & multiply by matrix product, sum to rhs shape" step in multi-headed attention, but for headless
    """
    lhs, lhs_dims = unsqueeze_multi(value, *(d + 1 for d in dims))
    values_scaled = lhs * scores
    values_summed = values_scaled.sum(tuple(d - 1 for d in lhs_dims))
    return values_summed


def softand(A: Tensor, B: Tensor):
    AB = torch.cat((A, B), 0)
    return (AB * ff.softmin(AB, 0)).sum(0)


def softor(A: Tensor, B: Tensor):
    AB = torch.cat((A, B), 0)
    return (AB * ff.softmax(AB, 0)).sum(0)


def softxor(A: Tensor, B: Tensor):
    AB = torch.cat((A, B), 0)
    C = (AB * ff.softmin(AB, 0)).sum(0)
    D = (AB * AB.softmax(0)).sum(0)
    return softand(C, D)
