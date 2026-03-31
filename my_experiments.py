# from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import functional as ff
from torch.nn.parameter import Parameter
from torch import Tensor

from typing import Callable, Iterable


NOT_EPSILON = 1
"""A number that isn't very small, and definitely not 0"""


def swishmax(
    input: Tensor, dim: torch.types._size | int = -1, *, shrink_factor=None
) -> Tensor:
    "shrink_factor is used to divide the xe^x before normalizing"
    xexp = input * torch.exp(input - torch.amax(input, dim=dim, keepdim=True))
    if shrink_factor is not None:
        xexp = xexp / shrink_factor
    out = torch.div(
        xexp, (torch.sum(torch.abs(xexp), dim=dim, keepdim=True) + NOT_EPSILON)
    )
    return out


def sillog(X: Tensor) -> Tensor:
    log_X = X.abs().log1p()
    return X.sigmoid() * log_X.where(X > 0, X)


def mask2d(x: Tensor) -> Tensor:
    return x.movedim(-1, 0).triu(diagonal=0).movedim(0, -1)


def subset_loss(input: Tensor, target: Tensor):
    """loss function approaching zero as input approaches being a subset of target"""


def make_weight(*dims: int):
    return Parameter(torch.randn(dims))


class FractalTransformer(nn.Module):
    def __init__(
        self,
        dims: int,
        depth: int,
        module_1d: Callable[[int], nn.Module],
        *,
        mask: Callable[[Tensor], Tensor] = nn.Identity(),
        pos_encoder: Callable[[Tensor, Tensor], Tensor] = nn.Identity(),
        attention_1d: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.dims = dims
        self.depth = depth
        self.module_1d = module_1d
        self.mask = mask
        self.pos_encoder = pos_encoder

        self.pre = self.recurse(dims, depth - 1)
        self.mid = (
            FractalAttention(dims, depth, mask=self.mask)
            if attention_1d is None
            else FractalAttention(
                dims, depth, mask=self.mask, attention_1d=attention_1d
            )
        )
        self.end = self.recurse(dims, depth - 1)

    def recurse(self, dims, depth):
        if depth >= 2:
            return FractalTransformer(dims, depth, self.module_1d, mask=self.mask)
        else:
            return self.module_1d(dims)

    def forward(self, X: Tensor):
        X = self.pre(X)
        X = X + self.mid(
            self.pos_encoder(
                X,
                torch.arange(X.shape[-2], device=X.device)
                .broadcast_to(X.shape[:-1])
                .unsqueeze(-1),
            )
        )
        X = self.end(X)
        return X


class FractalAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        depth: int,
        *,
        mask: Callable[[Tensor], Tensor] = nn.Identity(),
        attention_1d: Callable[[int], nn.Module] = lambda dims: nn.Linear(
            dims, dims, False
        ),
    ) -> None:
        super().__init__()
        self.dims = dims
        self.depth = depth
        assert depth >= 1
        (
            self.to_key,
            self.to_query,
            self.to_value_attention,
            self.to_attention_logits,
            self.to_value_out,
        ) = (
            (attention_1d(dims) for _ in range(5))
            if depth == 2
            else (
                FractalAttention(dims, depth - 1, mask=mask, attention_1d=attention_1d)
                for _ in range(5)
            )
        )
        self.row_dims = tuple(-3 - i for i in range(0, self.depth - 1))
        self.col_dims = tuple(-2 - i for i in range(0, self.depth - 1))

        self.mask = mask

    def batchify(self, A: Tensor, lhs: bool, current_depth=None):
        if current_depth is None:
            current_depth = self.depth
        if current_depth == 1:
            return A
        else:
            return self.batchify(
                A.unsqueeze((0 if lhs else -1) - current_depth),
                lhs,
                current_depth - 1,
            )

    def forward(
        self,
        query_tokens: Tensor,
        key_tokens: Tensor | None = None,
        value_tokens: Tensor | None = None,
        *,
        return_attention=False,
    ):
        if key_tokens is None:
            key_tokens = query_tokens

        if value_tokens is None:
            value_tokens = key_tokens
        pass

        query = query_tokens + self.to_query(query_tokens)
        # [..., Q, ...]
        key = key_tokens + self.to_key(key_tokens)
        # [..., K, ...]

        attention_raw = self.mask(
            self.batchify(key, True) * self.batchify(query, False)
        )
        # [..., K, Q, ..., T] = [..., K, 1, ..., T] * [..., 1, Q, ..., T]

        attention_logits = attention_raw + self.to_attention_logits(attention_raw)
        # [..., K, Q, ..., T]

        attention_scale = swishmax(attention_logits, dim=self.row_dims)
        # [...,^K, Q, ..., T]

        value = value_tokens + self.to_value_attention(value_tokens)
        # [..., K, ...]
        values_scaled = self.batchify(value, True) * attention_scale
        # [..., K, Q, ...] = [..., K, 1, ...] * [..., K, Q, ...]

        value_sum = values_scaled.sum(self.row_dims)
        # [..., Q, T]
        value_out = value_sum + self.to_value_out(value_sum)
        # [..., Q, T]

        # print(self.depth, end="")
        if return_attention:
            return value_out, attention_scale, attention_raw
        else:
            return value_out


class AttentionDeduplicate(nn.Module):
    def __init__(
        self,
        dims: int,
        mask: nn.Module = nn.Identity(),
        *,
        key_bias=False,
        query_bias=False,
        value_attention_bias=False,
        attention_logits_bias=False,
        value_out_bias=False,
    ) -> None:
        super().__init__()

        # fmt:off
        self.to_key              = nn.Linear(dims, dims, key_bias)
        self.to_query            = nn.Linear(dims, dims, query_bias)
        self.to_value_attention  = nn.Linear(dims, dims, value_attention_bias)
        self.to_attention_logits = nn.Linear(dims, dims, attention_logits_bias)
        self.to_value_out        = nn.Linear(dims, dims, value_out_bias)
        # fmt:on

        self.mask = mask

    def forward(
        self,
        query_tokens: Tensor,
        key_tokens: Tensor | None = None,
    ):
        """
        number of queries:  Q
        length of queries:  QT
        number of keys:     K
        length of keys:     KT

        query_tokens.shape  [..., Q, T]
        key_tokens.shape    [..., K, T]

        batching and broadcasting for dimensions prior to last two
        """

        if key_tokens is None:
            key_tokens = query_tokens

        key: Tensor = self.to_key(key_tokens)
        # [..., K, T]
        query: Tensor = self.to_query(query_tokens)
        # [..., Q, T]

        key_similarity = (
            ff.cosine_similarity(key.unsqueeze(-2), key.unsqueeze(-3), dim=-1)
            # [..., K, K]
            .pow(2)
            .sum(-1, keepdim=True)
            # [..., K, 1]
            .unsqueeze(-1)
            # [..., K, 1, 1]
        )

        attention_raw: Tensor = self.mask(key.unsqueeze(-2) * query.unsqueeze(-3))
        # [..., K, Q, T] = [..., K, 1, T] * [..., 1, Q, T]
        attention_logits: Tensor = self.to_attention_logits(attention_raw)
        # [..., K, Q, T]
        attention_scale = swishmax(
            attention_logits,
            dim=-2,
            shrink_factor=key_similarity,
        )
        # [..., K, Q, T]

        value_raw: Tensor = self.to_value_attention(key)
        # [..., K, T]
        values_scaled = value_raw.unsqueeze(-2) * attention_scale
        # [..., K, Q, T] = [..., K, 1, T] * [..., K, Q, T]
        value_sum = values_scaled.sum(-3)
        # [..., Q, T]
        value_out: Tensor = self.to_value_out(value_sum)
        # [..., Q, T]

        return value_out


class HeadlessAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        *,
        mask: Callable[[Tensor], Tensor] = nn.Identity(),
        module_factory: Callable[[int], nn.Module] = (
            lambda dims: nn.Linear(dims, dims, False)
        ),
    ) -> None:
        super().__init__()

        (
            self.to_key,
            self.to_query,
            self.to_value_attention,
            self.to_attention_logits,
            self.to_value_out,
        ) = (module_factory(dims) for _ in range(5))

        self.mask = mask

    def forward(
        self,
        query_tokens: Tensor,
        key_tokens: Tensor | None = None,
        value_tokens: Tensor | None = None,
    ):
        """
        number of queries:  Q
        length of queries:  QT
        number of keys:     K
        length of keys:     KT

        query_tokens.shape  [..., Q, T]
        key_tokens.shape    [..., K, T]

        batching and broadcasting for dimensions prior to last two
        """

        if key_tokens is None:
            key_tokens = query_tokens

        if value_tokens is None:
            value_tokens = key_tokens

        key: Tensor = self.to_key(key_tokens)
        # [..., K, T]
        query: Tensor = self.to_query(query_tokens)
        # [..., Q, T]

        attention_raw: Tensor = key.unsqueeze(-2) * query.unsqueeze(-3)
        # [..., K, Q, T] = [..., K, 1, T] * [..., 1, Q, T]
        attention_logits: Tensor = self.mask(self.to_attention_logits(attention_raw))
        # [..., K, Q, T]
        attention_scale = swishmax(attention_logits, dim=-2)
        # [..., K, Q, T]

        value: Tensor = self.to_value_attention(value_tokens)
        # [..., K, T]
        values_scaled = value.unsqueeze(-2) * attention_scale
        # [..., K, Q, T] = [..., K, 1, T] * [..., K, Q, T]
        value_sum = values_scaled.sum(-3)
        # [..., Q, T]
        value_out: Tensor = self.to_value_out(value_sum)
        # [..., Q, T]

        return value_out


class AttentionZP(nn.Module):
    # """One query vector per operation"""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        heads: int,
        attention_size: int,
        compress_size: int,
        bias_query: bool,
        return_attention: bool = False,
        mask: Callable[[Tensor], Tensor] | None = None,
        dropout=0.0,
    ) -> None:
        super().__init__()

        # should preserve 0s
        self.key_down = make_weight(heads, key_size, attention_size)

        # should preserve 0s
        self.query_down = make_weight(heads, query_size, attention_size)
        # should NOT preserve 0s
        self.bias_query = bias_query
        if bias_query:
            self.query_down_bias = make_weight(heads, 1, attention_size)

        # should preserve 0s
        self.compress = query_size > compress_size * 2
        if self.compress:
            self.value_down = make_weight(heads, query_size, compress_size)
            self.value_up = make_weight(heads, compress_size, query_size)
        else:
            self.value_over = make_weight(heads, query_size, query_size)

        self.return_attention = return_attention
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: Tensor,
        key_tokens: Tensor | None = None,
    ):
        """
        number of queries:  Q
        length of queries:  QT
        number of keys:     K
        length of keys:     KT
        number of heads:    H

        query_tokens.shape  [..., Q, T]
        key_tokens.shape    [..., K, T]

        batching and broadcasting for dimensions prior to last two
        """

        if key_tokens is None:
            key_tokens = query_tokens

        key_tokens = key_tokens.unsqueeze(-3)
        # [..., 1, K, KT]

        query_tokens = query_tokens.unsqueeze(-3)
        # [..., 1, Q, QT]

        key = key_tokens @ self.key_down
        # [..., H, K, A]

        query = query_tokens @ self.query_down
        if self.bias_query:
            query = query + self.query_down_bias
        # [..., H, Q, A]

        attention_logits = key @ query.transpose(-2, -1)
        # [..., H, K, Q]

        attention_dist_keys = swishmax(attention_logits, dim=-2)
        # [..., H, K, Q]

        values_scaled = key_tokens.unsqueeze(-2) * attention_dist_keys.unsqueeze(-1)
        # [..., H, K, Q, T] = [..., 1, K, 1, KT] * [..., H, K, Q, 1]

        value_shift = values_scaled.sum(-3)
        # [..., H, Q, T]

        if self.compress:
            value_shift @= self.value_down
            # [..., H, Q, A]
            value_shift @= self.value_up
            # [..., H, Q, T]
        else:
            value_shift @= self.value_over
            # [..., H, Q, T]

        value_shift = value_shift.sum(-3)
        #  [..., Q, T]

        # if self.return_attention:
        #     return value_shift, attention_logits
        # else:
        return self.dropout(value_shift)


class LinearActivateZP(nn.Module):
    """
    Linear with bias but it's a zero preserving unary function
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        self.bias = nn.Parameter(torch.randn(out_features))
        self.linear = nn.Linear(in_features, out_features, False)
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:

        return self.activation(self.linear(input) + self.bias) - self.activation(
            self.bias
        )


class MultiplyPair(nn.Module):
    def __init__(
        self,
        A: Callable[[Tensor], Tensor],
        B: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.a = A
        self.b = B

    def forward(self, X):
        return self.a(X) * self.b(X)


class Swishmoid(nn.Module):
    def __init__(
        self,
        dims: int,
        factors: int,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(dims, factors))
        self.scale = nn.Parameter(torch.randn(dims, factors))
        self.factors = factors

    def forward(self, X: Tensor) -> Tensor:
        return ff.tanh(X) * torch.prod(
            ff.tanh(X.unsqueeze(-1) * self.scale) + self.bias, dim=-1
        )


class RotPosEncode(nn.Module):
    def __init__(
        self,
        dims: int,
        position_dims: int,
        min_wavelength: float,
        max_wavelength: float,
    ) -> None:
        super().__init__()

        self.position_dims = position_dims

        self.features_per_step = 2 * position_dims

        self.steps = dims // self.features_per_step

        self.unused_dims = dims - self.steps * self.features_per_step

        self.last_pos = nn.Buffer()
        self.sin_factor = nn.Buffer()
        self.cos_factor = nn.Buffer()

        i_range = torch.arange(0, self.steps)
        # [steps]

        i_mult = (
            (2 * math.pi)
            * ((max_wavelength / min_wavelength) ** (i_range / (self.steps - 1)))
            / max_wavelength
        )
        self.scale_sequence = nn.Buffer(i_mult)
        # [steps]

    def forward(self, X: Tensor, pos: Tensor | None) -> Tensor:
        """
        Argument shapes:
        ```
        X   = [B..., B, T]
        pos = [B..., B, pos_dims]
        """
        out = torch.empty_like(X)
        # [B..., T]

        view_shape = list(out.shape)
        view_shape.pop()
        view_shape.append(self.steps)
        view_shape.append(self.position_dims)
        view_shape.append(2)
        # [B..., steps, pos_dims, 2]

        X_view = X[..., self.unused_dims :].view(view_shape)
        # [B..., steps, pos_dims, 2]

        out_view = out[..., self.unused_dims :].view(view_shape)
        # [B..., steps, pos_dims, 2]

        if pos is not None:  # nested for type check
            if self.last_pos is None or pos.equal(self.last_pos):
                self.last_pos = pos
                # [B..., pos_dims]

                pos_view = pos.unsqueeze(-2)
                # [B..., 1, pos_dims]

                self.sin_factor = torch.sin(
                    pos_view * self.scale_sequence.unsqueeze(-1)
                )
                self.cos_factor = torch.cos(
                    pos_view * self.scale_sequence.unsqueeze(-1)
                )
                # [B..., steps, pos_dims] = [B..., 1, pos_dims] * [steps, 1]
        # print(pos.shape, self.cos_factor.shape)

        out_view[..., 0] = (
            X_view[..., 0] * self.cos_factor + X_view[..., 1] * self.sin_factor
        )
        out_view[..., 1] = (
            X_view[..., 1] * self.cos_factor - X_view[..., 0] * self.sin_factor
        )

        out[..., : self.unused_dims] = X[..., : self.unused_dims]

        return out


class SineEncoding(torch.nn.Module):
    def __init__(
        self,
        dim_pairs: int,
        # scale_divisor,
        # exp_factor=1.0,
        min_wavelength: float,
        max_wavelength: float,
    ) -> None:
        super().__init__()
        self.D = dim_pairs
        i_range = torch.arange(0, self.D)
        i_mult = (
            (2 * math.pi)
            * ((max_wavelength / min_wavelength) ** (i_range / (dim_pairs - 1)))
            / max_wavelength
        )
        self.I = torch.nn.Buffer(i_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        out_shape = list(x.size())
        out_shape[-1] = self.D * 2
        out = torch.zeros(out_shape)
        out[..., 0::2] = torch.sin(x * self.I)
        out[..., 1::2] = torch.cos(x * self.I)
        return out


class ConvNDAttention(nn.Module):
    """Channel last convention"""

    def __init__(
        self,
        conv_dims: Iterable[int],
        radius: Iterable[int],
        step: Iterable[int],
        attention: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        """`conv_dims` must be all negative"""
        super().__init__()
        self.conv_dims = tuple(conv_dims)
        assert all((i < 0) for i in self.conv_dims)
        self.radius = radius
        self.step = step
        # self.conv_dims = (self.conv_dims[i] - i for i in range(len(self.conv_dims)))
        self.attention = attention

    def forward(self, X: Tensor):
        original_shape = X.size()
        for i, r, s in self.conv_dims, self.radius, self.step:
            X = X.unfold(i, 1 + 2 * r, s)
            X = X.movedim(-1, 0)
        X = X.reshape([-1] + list(X.shape[: -len(original_shape)]))
        cell_count = X.shape[-1]
        center = cell_count // 2

        query = X[center : center + 1].movedim(0, -2)
        key = torch.cat((X[:center], X[center + 1 :])).movedim(0, -2)

        return self.attention(query, key).squeeze(-2)


class Conv2DAttention(nn.Module):
    # def __init__(self, *args, ) -> None:
    #     super().__init__(*args, )
    def __init__(self, attention_block: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.attention = attention_block
        self.pad = nn.CircularPad2d(3 // 2)

    def forward(self, image: Tensor) -> Tensor:
        width = 3

        image = self.pad(image)

        sections = image.unfold(-2, width, 1).unfold(-2, width, 1)
        sections = sections.reshape(list(sections.shape[:-2]) + [-1])
        middle = sections.shape[-1] // 2

        # print(middle)

        # print(sections.shape)

        keys = torch.cat((sections[..., :middle], sections[..., middle + 1 :]), dim=-1)
        # print(keys.shape, "K")

        query = sections[..., middle : middle + 1]
        # print(query.shape, "Q")

        keys = keys.movedim(1, -1)
        query = query.movedim(1, -1)

        # print(keys,query)

        outs, _ = self.attention.forward(query, keys)

        return outs.movedim(-1, 1)
