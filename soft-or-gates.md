---
tags:
 - softmax
---


# Soft OR Gates: Ternary Logic in Transformer MLPs

## 1. Ternary Logic Foundation

### 1.1 Multiplication as XNOR

Define ternary values over $\{-1, 0, +1\}$:
- $-1$ = false
- $0$ = unknown / irrelevant
- $+1$ = true

For scalars $a, b \in \{-1, 0, +1\}$:

| $a$ | $b$ | $a \cdot b$ | XNOR$(a,b)$ |
|-----|-----|-------------|-------------|
| -1  | -1  | +1          | true (both false) |
| -1  | +1  | -1          | false (mismatch) |
| +1  | -1  | -1          | false (mismatch) |
| +1  | +1  | +1          | true (both true) |
| 0   | any | 0           | unknown |

So scalar multiplication over $\{-1, 0, +1\}$ is **exactly the XNOR operation** extended with zero for "unknown." Two matching signs produce +1 (agreement); two opposing signs produce −1 (disagreement); any zero produces 0 (don't care).

The vector dot product is then:

$$
Q_i \cdot K_j = \sum_k Q_{ik} \cdot K_{jk}
$$

which accumulates XNOR agreements across all dimensions — a total agreement score.

### 1.2 Softmax as Differentiable OR / MAX

The softmax function, as temperature approaches zero, becomes an argmax selector (hardmax):

$$
\lim_{T \to 0} \operatorname{softmax}(x/T) = \operatorname{hardmax}(x)
$$

The operation:

$$
\operatorname{out}(x) = \sum_i x_i \cdot \operatorname{softmax}(x)_i
$$

approaches $\max(x)$ as $T \to 0$.

In ternary logic over $\{-1, 0, +1\}$:

$$
\max(a, b) = \text{OR}(a, b)
$$

For any number of inputs, $\max$ is the multi-way OR.

**Truth table** for 2-element softmax MAX at $T=0$:

| $a$ | $b$ | $\max(a,b)$ | $\sum x \cdot \text{softmax}_0(x)$ | Ternary OR |
|-----|-----|-------------|-----------------------------------|------------|
| -1  | -1  | -1          | $(-1)(0.5)+(-1)(0.5) = -1$       | $\lnot a \land \lnot b$ |
| -1  | 0   | 0           | $(-1)(0)+(0)(1) = 0$             | unknown |
| -1  | +1  | +1          | $(-1)(0)+(1)(1) = +1$            | true |
| 0   | -1  | 0           | $(0)(1)+(-1)(0) = 0$             | unknown |
| 0   | 0   | 0           | $(0)(0.5)+(0)(0.5) = 0$          | unknown |
| 0   | +1  | +1          | $(0)(0)+(1)(1) = +1$             | true |
| +1  | -1  | +1          | $(1)(1)+(-1)(0) = +1$            | true |
| +1  | 0   | +1          | $(1)(1)+(0)(0) = +1$             | true |
| +1  | +1  | +1          | $(1)(0.5)+(1)(0.5) = +1$         | true |

Every row matches. $\sum_i x_i \operatorname{softmax}_0(x)_i = \max(x) = \text{OR}(x)$.

### 1.3 NOT and AND via Negation

NOT is negation:

$$
\text{NOT}(a) = -a
$$

De Morgan's laws hold in this ternary system:

$$
\begin{aligned}
\text{AND}(a, b) &= \text{NOT}(\text{OR}(\text{NOT}(a), \text{NOT}(b))) \\
&= -\max(-a, -b) \\
&= \min(a, b)
\end{aligned}
$$

Since learned linear projections absorb sign flips (multiplying by a learned negative weight is equivalent to negating that input), the network can construct any binary logic gate for any dimension:

| Gate | Real expression | Ternary interpretation |
|------|----------------|----------------------|
| OR(a,b) | $\max(a,b)$ | $a \lor b$ |
| AND(a,b) | $\min(a,b)$ | $a \land b$ |
| a AND NOT b | $\min(a, -b)$ | $a \land \lnot b$ |
| NOT a AND b | $\min(-a, b)$ | $\lnot a \land b$ |
| NOR(a,b) | $-\max(a,b)$ | $\lnot(a \lor b)$ |
| NAND(a,b) | $-\min(a,b)$ | $\lnot(a \land b)$ |

## 2. Replacing SwiGLU with Softmax-Pair MAX

### 2.1 Standard SwiGLU

The SwiGLU variant of the MLP (used in LLaMA, PaLM, etc.):

```python
# SwiGLU MLP
def mlp_swiglu(x):
    gate = F.silu(W_gate(x))     # smooth {0,1} gate
    up   = W_up(x)               # values to gate
    out  = W_down(gate * up)     # elementwise gating
    return out
```

SiLU is a smooth approximation of $x \cdot \text{step}(x)$, a binary $(0/1)$ gate. This is a smooth binary AND where the gate determines whether the value passes.

### 2.2 Ternary Softmax-Pair MAX Gate

Replaces SiLU-based gating with a literal ternary OR operation:

```python
# Soft OR MLP
def mlp_soft_or(x):
    A = W_A(x)      # first projection → d_ff
    B = W_B(x)      # second projection → d_ff
    
    # For each pair (A_i, B_i):
    #   out_i = sum([A_i, B_i] · softmax([A_i, B_i]))
    #         = max(A_i, B_i)  (differentiable OR)
    logits = torch.stack([A, B], dim=-1)  # [batch, seq, d_ff, 2]
    weights = F.softmax(logits, dim=-1)   # softmax over the 2-element dimension
    out = (torch.stack([A, B], dim=-1) * weights).sum(dim=-1)  # [batch, seq, d_ff]
    
    out = W_down(out)  # project back to d_model
    return out
```

The operation in one expression:

$$
\text{out}_i = \sum_{j} A_{ij} \cdot \text{softmax}([A_{ij}, B_{ij}])_0 + B_{ij} \cdot \text{softmax}([A_{ij}, B_{ij}])_1
$$

where $j$ indexes the hidden dimension.

At $T=0$:

$$
\text{out}_i = \max(A_i, B_i)
$$

which is exactly the ternary OR gate applied elementwise across the hidden dimension.

### 2.3 Why This Matters

SwiGLU uses a smooth binary gate (SiLU) that produces a value in $(0, 1)$. This is a continuous approximation of "let the value through or block it."

The softmax-pair MAX gate operates on a fundamentally different principle:
- Both $A$ and $B$ carry information (content + CoPE phases if applicable)
- The gate selects between them based on which has the larger real value
- This is a ternary OR over latent predicates, not a binary "on/off" switch
- When the vectors are complex-valued (as in CoPE), the selected element carries its CoPE phase through

### 2.4 AND via Learned Negation

Since AND = $-\max(-A, -B) = \min(A, B)$, the network can implement AND simply by:

1. Learning negative weights in $W_A$ or $W_B$ to effectively negate certain inputs
2. Applying the softmax MAX (which is OR)
3. Negating the output (if needed)

No additional mechanism is required — the existing linear projections handle it.

### 2.5 Temperature Schedule

In practice, the softmax temperature can be learned or scheduled:

```python
# With learnable temperature
def soft_or_gate(A, B, temperature=1.0):
    logits = torch.stack([A / temperature, B / temperature], dim=-1)
    weights = F.softmax(logits, dim=-1)
    return (torch.stack([A, B], dim=-1) * weights).sum(dim=-1)
```

A low temperature makes the gate behave more like hard OR/MAX. A high temperature keeps it smooth and differentiable. The temperature can be a learned scalar per layer.

## 3. Summary

- **Ternary values** $\{-1, 0, +1\}$: $-1$ = false, $0$ = unknown, $+1$ = true
- **Dot product** = accumulated XNOR = agreement score
- **Softmax** = differentiable argmax
- **$\sum x \cdot \text{softmax}(x)$** = differentiable MAX = differentiable OR
- **NOT** = negation
- **AND** = $-\max(-A, -B) = \min(A, B)$ via De Morgan
- **Replaces SwiGLU** with a pair of projections gated by softmax-pair MAX

---

## See Also

- [CoPE-SoftOR MLP](cope-softor-mlp.md) — CoPE integration and MLP composition details
- [Associative Memory View](associative-memory-view.md) — Attention as a differentiable associative array (ternary interpretation)
- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan, including Design Decision 2 (MLP architecture)
