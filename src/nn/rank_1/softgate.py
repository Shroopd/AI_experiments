from __future__ import annotations

from torch.nn import Module, Linear

import src.functional as xff


class SoftGate(Module):
    def __init__(
        self,
        input: int,
        middle: int,
        output: int | None,
        *,
        bias=False
    ) -> None:
        super().__init__()
        self.linA = Linear(input, middle,bias)
        self.linB = Linear(input, middle,bias)
        if output is not None:
            self.linout = Linear(middle, output)

    def forward(self, X):
        A = self.linA(X)
        B = self.linB(X)
        AorB = xff.softor(A, B)

        return AorB if self.linout is None else self.linout(AorB)