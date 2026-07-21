# Attention as a Differentiable Associative Array

**Crossover topic:** This file documents the associative-array (dictionary) interpretation of attention, which is foundational to both [Complex Positional Encoding (CoPE)](cope.md) and [Soft OR Gates](soft-or-gates.md).

---

## The Associative Array Model

Standard attention is usually described as a "similarity search" between queries and keys. Under the associative-array interpretation, that description is misleading.

An associative array (dictionary) has three roles:

| Role | In a dictionary | In attention |
|------|----------------|--------------|
| Address | Hash/Index | Key |
| Lookup request | Key argument | Query |
| Payload | Stored value | Value |

The dot product $Q_i \cdot K_j$ is not measuring geometric similarity. It is determining **whether the address matches the lookup request**. Each dimension of $K$ and $Q$ acts as a fuzzy ternary flag:

- **Positive (+1)** → satisfies this criterion
- **Negative (−1)** → contradicts this criterion
- **Zero (0)** → irrelevant for this criterion

The dot product accumulates agreement evidence across all dimensions. Softmax then acts as a differentiable winner-take-all: it selects which stored entry best matches the query and retrieves its value.

---

## How CoPE Extends This View

Attention can be viewed as a differentiable associative array:

- Keys act as addresses.
- Queries act as lookup requests.
- Values act as stored content.

Standard attention computes soft retrieval:

$$
\mathrm{softmax}(QK^T)V
$$

CoPE extends this view by making addresses themselves algebraically manipulable.

Instead of only asking:

> Which stored value matches this address?

the model can compute:

> What address results from combining these addresses?

This turns positional information from a passive feature into an active computational structure.

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [Soft OR Gates](soft-or-gates.md) — Ternary logic foundation
- [CoPE-SoftOR MLP](cope-softor-mlp.md) — CoPE composition inside the MLP
- [Gated Attention](gated-attention.md) — Gated attention mechanism
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan
