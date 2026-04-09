import math
import itertools
from itertools import product

import numpy

dims = 3

foo = product((-1, 0, 1), repeat=dims)

buckets = [[] for _ in range(dims + 1)]


def dirs(X):
    return sum(abs(i) for i in X)


for v in foo:
    buckets[dirs(v)].append(v)

for bi in range(len(buckets)):
    b = buckets[bi]
    b = list(1 + numpy.array(b).T)
    for i in range(len(b)):
        b[i] = tuple(b[i])
    buckets[bi] = tuple(b)  # type: ignore


bar = numpy.zeros((3,) * dims, dtype=int)
for d in range(dims):
    numpy.swapaxes(bar, d, -1)[:] += numpy.arange(3) * 10 ** (dims - d - 1)

print(bar)
for b in buckets:
    # print(bar[tuple(1 + numpy.array(b).T)])
    print(bar[b])
