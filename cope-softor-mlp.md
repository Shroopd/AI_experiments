# CoPE Composition Inside the MLP via Soft OR Gates

**Crossover topic:** This file documents how CoPE phase composition is realized within the modified MLP using Soft OR gates. It belongs to both [Complex Positional Encoding (CoPE)](cope.md) and [Soft OR Gates](soft-or-gates.md).

---

## CoPE Composition Inside the MLP

CoPE composition does not happen inside attention. It happens inside a **modified MLP** (the feed-forward network), using the Soft OR gating mechanism.

### Mechanism

The embedding $h$ contains both content features (ternary predicates) and CoPE phases in different subspaces of the same vector. The MLP projects two (or more) distinct subspaces and composes their CoPE phases through elementwise operations:

```python
def mlp_with_cope_composition(x):
    A = W_A(x)          # first projection — extracts one CoPE-carrying subspace
    B = W_B(x)          # second projection — extracts another CoPE-carrying subspace
    V = A * B           # Hadamard product: composes CoPE phases (A⊙B = phase addition)
    # The Soft OR gate selects between A and B (or between V and something else)
    # based on which has the greater real-part relevance
    return W_down(V)
```

### Why the MLP

- Attention operates across tokens (it answers "which other token should I look at?")
- The MLP operates per-token (it answers "what computation should I perform on what I've gathered?")

CoPE composition is a per-token computation: "given the quantities I've retrieved through attention, combine them arithmetically." This naturally belongs in the MLP, not in the attention mechanism.

After attention retrieves values from other tokens (which may carry CoPE phases encoded in their value subspaces), the MLP operates on the aggregated embedding and can compose those phases.

### Four-Operation Arithmetic

The four Hadamard-product operations form a complete address arithmetic:

| Operation | Expression | Meaning |
|-----------|-----------|---------|
| Forward addition | $A \odot B$ | $a + b$ |
| Forward subtraction | $A \odot \overline{B}$ | $a - b$ |
| Reverse subtraction | $\overline{A} \odot B$ | $b - a$ |
| Negated addition | $\overline{A} \odot \overline{B}$ | $-(a + b)$ |

The Soft OR gate selects which composition result enters the residual stream.

---

## Integration with CoPE

When CoPE phases are present in the embedding ($h_i = \text{embed(token}_i) \odot e^{ip\omega}$), the projections $W_A$ and $W_B$ extract CoPE-carrying subspaces. Their Hadamard product (elementwise multiplication) composes the CoPE phases:

$$
(A \odot B)_j = \left(\sum_k h_k W^A_{kj}\right) \cdot \left(\sum_l h_l W^B_{lj}\right)
$$

Each projection is a complex-weighted combination of the CoPE phases. The elementwise product adds the phases.

The Soft OR gate does not directly compose phases — it selects between $A$ and $B$ based on their real parts — but it determines *which* composition enters the residual stream. The model can use one branch to compose phases and the other to provide an alternative, with the gate selecting the appropriate result.

---

## Computational Flow in Context

```
embedding h  (if CoPE: h = embed(token) ⊙ CoPE(pos))
    │
    ├──► W_Q → Q  (for attention)
    ├──► W_K → K  (for attention)
    ├──► W_V → V  (for attention)
    │
    └──► W_A → A ──┐
         W_B → B ──┤
                   ▼
              softmax-pair MAX
                   │
                   ▼
              W_down → h_mlp  (added to residual)
```

Attention remains standard. CoPE composition and ternary gating happen inside the MLP. The embedding is a unified object — content and addresses coexist in the same vector.

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [Soft OR Gates](soft-or-gates.md) — Ternary logic foundation and SwiGLU replacement
- [Associative Memory View](associative-memory-view.md) — Attention as differentiable associative array
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan, including Design Decision 2 (MLP architecture)
