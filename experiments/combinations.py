import math
import itertools
from itertools import product

import numpy

dims = 4
radius = 1

foo = product(range(-radius, radius + 1), repeat=dims)

buckets = [[] for _ in range(dims + 1)]


def dirs(X):
    return dims - sum(1 if (i == 0) else 0 for i in X)


for v in foo:
    buckets[dirs(v)].append(v)

for bi in range(len(buckets)):
    b = buckets[bi]
    b = list(radius + numpy.array(b).T)
    for i in range(len(b)):
        b[i] = tuple(b[i])
    buckets[bi] = tuple(b)  # type: ignore


bar = numpy.zeros((radius * 2 + 1,) * dims, dtype=int)
for d in range(dims):
    numpy.swapaxes(bar, d, -1)[:] += numpy.arange(radius * 2 + 1) * 10 ** (dims - d - 1)

print(bar)
for i in range(len(buckets)):
    b = buckets[i]
    # print(bar[tuple(1 + numpy.array(b).T)])
    print(bar[b])
    bar[b] = i
print(bar)
