# Complex-Valued Gated Attention: CoPE and Gated Attention

**Crossover topic:** This file documents how CoPE and Gated Attention compose together. It belongs to both [Complex Positional Encoding (CoPE)](cope.md) and [Gated Attention](gated-attention.md).

---

## CoPE and Gated Attention Are Independent but Composible

CoPE and Gated Attention are complementary ideas:

- **CoPE** changes the **representation** (complex-valued embeddings with composable phases)
- **Gated Attention** changes the **attention mechanism** (per-dimension softmax with learned logit recombination)

They can be used separately or together. The benchmark plan tests each both independently and in combination.

---

## Complex-Valued Gated Attention

When CoPE is active, all five weight matrices are complex-valued. The attention score becomes:

$$
\text{score}_{ij} = \text{Re}\left( \sum_m \left( \sum_k P_{ij}^{(k)} W^L_{km} \right) \right)
$$

Wait — this needs careful treatment. The logit $L$ is complex-valued (complex matrix times complex vector). We need a real-valued score for the softmax. Options:

**Option A: Take real part before softmax**

```python
L = self.W_L(P)           # complex-valued
L_real = L.real            # real part only
A = F.softmax(L_real, dim=2)
output = (A * V.real.sum(...))  # ... complicated
```

**Option B: Take real part of the conjugate-weighted product**

This matches the theoretical framework where attention scores come from $\text{Re}(Q^* K)$:

```python
P = Q.unsqueeze(2) * K.unsqueeze(1)  # elementwise P_ij = Q_i · K_j (scalar complex multiplication)
# But wait, this isn't the same as the conjugate product...
```

For the cleanest complex version, the attention score should be the real part of the conjugate-weighted inner product:

$$
\text{score}_{ij} = \text{Re}\left( \sum_k \overline{Q}_{ik} W^L_{km} K_{jk} \right)
$$

or equivalently, the Logit matrix operates on the pairwise conjugate product.

The full complex gated attention with conjugate products is left as an experimental refinement. For initial experiments, real-valued gated attention (with real $W^Q$, $W^K$, $W^L$, $W^V$, $W^O$) is sufficient to test the mechanism.

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [Gated Attention](gated-attention.md) — Full gated attention mechanism
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan
