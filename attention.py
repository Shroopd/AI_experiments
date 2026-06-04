# from __future__ import annotations

# import math
# import itertools
# from typing import Callable, Iterable

# import torch
# from torch import nn
# from torch.nn import functional as ff
# from torch.nn.parameter import Parameter
# from torch import Tensor


# def sparse_swishmax(input: Tensor, dim: torch.types._size | int = -1) -> Tensor:
#     # matrix expected layout: torch.sparse_coo
#     if not input.is_sparse:
#         raise ValueError("Input tensor must be sparse!")
    
#     input = input.coalesce()


#     xexp = input * torch.exp(input - torch.amax(input, dim=dim, keepdim=True))
#     out = torch.div(xexp, (torch.sum(torch.abs(xexp), dim=dim, keepdim=True) + 1))
#     return out
