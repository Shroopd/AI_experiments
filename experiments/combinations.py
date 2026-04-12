import math
import itertools
from itertools import product

# import numpy
import torch


def generate_diagonal_indices(dims, radius):
    def dirs(X):
        return dims - sum(1 if (i == 0) else 0 for i in X)

    positions = (
        list(p) for p in itertools.product(range(-radius, radius + 1), repeat=dims)
    )

    pos_buckets = [[] for _ in range(dims + 1)]

    for v in positions:
        pos_buckets[dirs(v)].append(v)

    indice_buckets = tuple(
        tuple(tuple(b[r][c] + radius for r in range(len(b))) for c in range(len(b[0])))
        for b in pos_buckets
    )
    return indice_buckets


dims = 2
radius = 1

# bar = torch.zeros((radius * 2 + 1,) * dims)
# for d in range(dims):
#     torch.swapaxes(bar, d, -1)[:] += torch.arange(radius * 2 + 1) * 10 ** (dims - d - 1)

bar = torch.arange((radius * 2 + 1) ** dims * 5).reshape(
    (1, 5) + tuple(radius * 2 + 1 for _ in range(dims))
)


bar = bar[None, ...]


buckets = generate_diagonal_indices(dims, radius)

print("A\n", bar)
print("A\n", buckets)
print("A\n", bar.shape)
for i in range(len(buckets)):
    b = buckets[i]
    # print(bar[tuple(1 + numpy.array(b).T)])
    print(i, "B\n", bar[(...,) + b].mT)
    print(i, "B\n", bar[(...,) + b].mT.shape)
    bar[(...,) + b] = i
print("C\n", bar)
