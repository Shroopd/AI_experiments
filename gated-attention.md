# Gated Attention: Replacing Head Splitting with a Learned Logit Matrix

## 1. Standard Multi-Head Attention (Reference)

For reference, standard multi-head attention (Vaswani et al. 2017) with $h$ heads and model dimension $d_{\text{model}}$:

**Projections (3 matrices per attention layer):**
- $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — Query projection (per head, or combined)
- $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — Key projection
- $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ — Value projection
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$ — Output projection

**Per-head computation:**

$$
\begin{aligned}
Q_i &= X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
\end{aligned}
$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $d_k = d_v = d_{\text{model}} / h$.

**Combination:**

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W^O
$$

**Key property:** Dimensions within a head share a single scalar attention weight per (query, key) pair. The $d_k$ dimensions of a head are always attended to together with the same strength.

## 2. Gated Attention: The Proposal

### 2.1 The Five Transformations

| # | Name | Matrix | Role |
|---|------|--------|------|
| 1 | **Key** | $W^K \in \mathbb{C}^{d \times d}$ | Projects input into key space |
| 2 | **Query** | $W^Q \in \mathbb{C}^{d \times d}$ | Projects input into query space |
| 3 | **Logit** | $W^L \in \mathbb{C}^{d \times d}$ | Recombines elementwise products into logits |
| 4 | **Value** | $W^V \in \mathbb{C}^{d \times d}$ | Projects input into value space (no head split) |
| 5 | **Output** | $W^O \in \mathbb{C}^{d \times d}$ | Projects output back to model dimension |

(Use real-valued matrices if CoPE is not active; the structure is the same.)

### 2.2 Computational Flow

```
  X  (shape: [batch, seq_len, d_model])
  │
  ├──► Q = X · W^Q   [batch, seq, d]
  ├──► K = X · W^K   [batch, seq, d]
  ├──► V = X · W^V   [batch, seq, d]
  │
  │   Step 1: Pairwise elementwise product
  │
  ├──► P_{ij} = Q_i ⊙ K_j   [batch, seq, seq, d]
  │   Each dimension k: Q_{ik} · K_{jk}  (scalar × scalar ≅ ternary flag)
  │
  │   Step 2: Logit transformation
  │
  ├──► L = P · W^L   [batch, seq, seq, d]
  │   L_{ij}^{(m)} = sum_k P_{ij}^{(k)} · W^L_{km}
  │   The logit matrix learns which elementwise products to sum together
  │   to form each output logit dimension — effectively defining dynamic "heads."
  │
  │   Step 3: Per-dimension softmax over keys
  │
  ├──► A_{ij}^{(m)} = softmax over j of L_{ij}^{(m)}
  │   Each dimension m gets its own independent attention distribution
  │   over all key positions. No head grouping.
  │
  │   Step 4: Gated value aggregation
  │
  ├──► output_i = sum_j A_{ij} ⊙ V_j   [batch, seq, d]
  │   Each dimension of V is independently gated by the corresponding
  │   dimension of the attention weight. Different dimensions of the same
  │   token can attend to different source positions.
  │
  │   Step 5: Output projection
  │
  └──► out = output · W^O   [batch, seq, d]
       Added to residual stream.
```

### 2.3 Pseudocode

```python
class GatedAttention(nn.Module):
    """5-transformation attention: Key, Query, Logit, Value, Output."""
    
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_L = nn.Linear(d_model, d_model, bias=False)  # Logit matrix
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        # x: [batch, seq, d_model]
        Q = self.W_Q(x)  # [batch, seq, d]
        K = self.W_K(x)  # [batch, seq, d]
        V = self.W_V(x)  # [batch, seq, d]
        
        # Pairwise elementwise product
        P = Q.unsqueeze(2) * K.unsqueeze(1)  # [batch, seq, seq, d]
        
        # Logit transformation
        L = self.W_L(P)  # [batch, seq, seq, d]
        
        # Apply causal mask
        if mask is not None:
            L = L + mask.unsqueeze(-1)  # broadcast mask over last dim
        
        # Per-dimension softmax over keys
        A = F.softmax(L, dim=2)  # [batch, seq, seq, d]
        
        # Gated value aggregation (elementwise)
        output = (A * V.unsqueeze(1)).sum(dim=2)  # [batch, seq, d]
        
        # Output projection
        out = self.W_O(output)  # [batch, seq, d]
        return out
```

### 2.4 Key Differences from Standard Attention

| Property | Standard Multi-Head | Gated Attention |
|----------|--------------------|----------------|
| Head structure | Hard partition of dimensions | Emergent from W^L |
| Attention distributions per (query, key) pair | $h$ scalars | $d$ scalars |
| Value gating | Scalar × entire sub-vector | Per-dimension gating |
| Score computation | Dot product over $d_k$ | Learned recombination via $W^L$ |
| Q,K,V dimensionality | Split into $h \times d_k$ | Full $d_{\text{model}}$ each |

## 3. Relationship to Multi-Head Attention

### 3.1 Equivalence Under Block-Diagonal W^L

Standard multi-head attention with $h$ heads and $d_k = d_{\text{model}} / h$ is a **special case** of gated attention where $W^L$ takes a specific block-diagonal form.

Let $W^L$ be a block-diagonal matrix where each block is $d_k \times d_k$ and filled with ones:

$$
W^L = I_h \otimes \mathbf{1}_{d_k \times d_k} =
\begin{bmatrix}
1 & 1 & 0 & 0 & \cdots \\
1 & 1 & 0 & 0 & \cdots \\
0 & 0 & 1 & 1 & \cdots \\
0 & 0 & 1 & 1 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

Then for dimension $m$ in block $b$:

$$
L_{ij}^{(m)} = \sum_{k = b \cdot d_k}^{(b+1) \cdot d_k - 1} P_{ij}^{(k)} = \sum_{k = b \cdot d_k}^{(b+1) \cdot d_k - 1} Q_{ik} \cdot K_{jk}
$$

which is exactly the dot product within head $b$.

Since all $d_k$ dimensions in block $b$ receive the same logit, they all produce the same softmax distribution. Therefore:

$$
A_{ij}^{(m)} = A_{ij}^{(m')} \quad \text{for all } m, m' \text{ in the same block}
$$

And the gated value aggregation becomes:

$$
\text{output}_i^{(block\ b)} = \sum_j A_{ij}^{(b)} \cdot V_j^{(block\ b)}
$$

which is identical to the output of head $b$ in standard multi-head attention.

### 3.2 Beyond Block-Diagonal

The advantage of gated attention is that $W^L$ is not required to be block-diagonal. It can learn:

- **Overlapping heads:** dimensions partially participate in multiple groups
- **Fuzzy boundaries:** soft mixtures rather than hard partitions
- **Fine-grained attention:** each dimension attends independently, up to $d$ independent attention distributions
- **Permutation heads:** a permutation matrix $W^L$ gives $d$ heads of 1 dimension each (maximally fine-grained)

### 3.3 Sparsity-Expressiveness Tradeoff

The Logit matrix $W^L$ controls the sparsity-expressiveness tradeoff:

- **All-ones matrix** = 1 giant head (maximum sharing, minimum expressiveness)
- **Block-diagonal with $d_k \times d_k$ blocks** = $h$ heads, standard behavior
- **Permutation matrix** = $d$ single-dimension heads (maximum expressiveness, minimum sharing)

The model learns where to sit on this spectrum based on the task.

### 3.4 Example

With $d_{\text{model}} = 4$, $h = 2$, $d_k = 2$:

Standard multi-head attention has two heads, each computing a 2-dim dot product and gating a 2-dim value subvector.

Gated attention with block-diagonal $W^L$:

$$
W^L = \begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 \\
0 & 0 & 1 & 1
\end{bmatrix}
$$

produces:
- Dimensions 0-1 share logit = $Q_{i0}K_{j0} + Q_{i1}K_{j1}$ (head 0)
- Dimensions 2-3 share logit = $Q_{i2}K_{j2} + Q_{i3}K_{j3}$ (head 1)

Identical behavior to standard multi-head attention.

## 4. Verification Protocol

To verify that gated attention converges to standard multi-head attention under the right conditions:

1. Initialize $W^L$ to the block-diagonal all-ones structure
2. Freeze $W^L$ (no gradient updates)
3. Train the model and compare behavior to a standard multi-head attention model with the same head configuration
4. If behavior matches (same loss curves, same attention patterns), the implementation is correct
5. Then unfreeze $W^L$ and observe how it diverges

Expected outcome: With $W^L$ frozen to block-diagonal, the two models should produce identical behavior. Once $W^L$ is trainable, it may converge to the block-diagonal structure (if standard heads are optimal) or diverge into a more expressive configuration (if the task benefits from finer-grained attention).

## 5. Computational Considerations

The pairwise elementwise product $P$ has shape $[n, n, d_{\text{model}}]$, requiring $O(n^2 d)$ memory compared to standard attention's $O(n^2 h)$.

For reference, size of the score tensor:

| seq_len | d_model | Standard (h=8) | Gated Attention | Ratio |
|---------|---------|----------------|-----------------|-------|
| 1024    | 512     | 8 MB           | 2 GB            | 256×  |
| 2048    | 512     | 32 MB          | 8 GB            | 256×  |
| 4096    | 512     | 128 MB         | 32 GB           | 256×  |
| 8192    | 512     | 512 MB         | 128 GB          | 256×  |

(Assuming 4-byte floats; complex doubles would be 2× per element.)

The DGX Spark (128 GB unified memory) can handle sequences up to ~4096 with d_model=512 before hitting memory limits. For longer sequences, gradient checkpointing, reduced batch size, or the standard multi-head attention baseline should be used.

Potential optimizations (future work):
- Gradient checkpointing on $P$ (recompute during backward instead of storing)
- Fused kernel for $P \cdot W^L$ (avoid materializing $P$)
- Block-sparse $W^L$ that enforces a head-like structure while remaining differentiable

## 6. Summary

Gated attention replaces the hard head-partition of standard multi-head attention with a learned linear transformation $W^L$ (the Logit matrix). The per-head softmax becomes a per-dimension softmax, allowing each dimension of the model to independently attend to different source positions. Multi-head attention emerges as the special case where $W^L$ is block-diagonal with all-ones blocks. This provides a strictly more expressive attention mechanism while remaining fully differentiable and backward-compatible with standard training techniques.

---

## See Also

- [CoPE-Gated Attention](cope-gated-attention.md) — Complex-valued extension for CoPE compatibility
- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan
