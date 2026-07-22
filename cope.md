# Complex Positional Encoding (CoPE)

## Core Idea

CoPE (Complex Positional Encoding) treats positional encodings as native elements of a complex vector space rather than as real-valued vectors that simulate complex rotations.

The central idea is that positions are not merely features attached to tokens. They are algebraic objects that can be composed, compared, and manipulated directly.

A position $p$ is represented as:

$$
\mathrm{CoPE}(p) = e^{ip\omega}
$$

where:
- $p$ is the position or quantity being encoded.
- $\omega$ is a vector of frequencies.
- The exponential is applied elementwise.

Each dimension represents a different rotational frequency.

### Embedding Integration

The CoPE phase is combined with the token embedding via **elementwise multiplication**:

$$
h_i = \text{embed(token}_i) \odot \mathrm{CoPE}(i)
$$

This preserves the complex phase structure through all projections (K, Q, V), matching the associative-array interpretation where positional information is an address that modulates content. This approach inherits the key advantage of RoPE while keeping the representation explicitly complex.

---

## Algebraic Properties

The defining property of CoPE is that positional composition becomes complex multiplication.

For two positions $a$ and $b$:

$$
\mathrm{CoPE}(a)\odot\mathrm{CoPE}(b)
=
e^{ia\omega}\odot e^{ib\omega}
=
e^{i(a+b)\omega}
=
\mathrm{CoPE}(a+b)
$$

where $\odot$ is elementwise (Hadamard) multiplication.

Therefore:

$$
\mathrm{CoPE}(a)\odot\mathrm{CoPE}(b)=\mathrm{CoPE}(a+b)
$$

Addition of encoded quantities becomes multiplication in representation space.

Using complex conjugation:

$$
\overline{\mathrm{CoPE}(b)}
=
e^{-ib\omega}
$$

so:

$$
\mathrm{CoPE}(a)\odot\overline{\mathrm{CoPE}(b)}
=
e^{i(a-b)\omega}
$$

Therefore:

$$
\mathrm{CoPE}(a)\odot\overline{\mathrm{CoPE}(b)}
=
\mathrm{CoPE}(a-b)
$$

Subtraction of encoded quantities becomes multiplication by the inverse element.

---

## Complex Weight Matrices Required

CoPE requires complex-valued weight matrices. RoPE achieves its rotational structure by hard-coding 2×2 rotation blocks in its linear transformations (pairing dimensions and coupling their parameters as $[[\cos\theta, -\sin\theta], [\sin\theta, \cos\theta]]$). This couples the real-pair representation of complex multiplication into the weight matrices themselves.

Without complex matrices, CoPE's complex-valued embeddings would have their real and imaginary parts transformed independently by real weights, losing the ability to hard-code rotational offsets. Complex weight matrices are necessary for CoPE to fully exploit its native complex representation.

See the **Z-shaped architecture comparison** in [Benchmark Survey Plan](benchmark-survey-plan.md) for the parameter-count implications.

---

## Relationship to RoPE

RoPE (Rotary Positional Embeddings) already uses the same mathematical structure, but represents it indirectly.

RoPE splits a real vector into 2D pairs:

$$
(x,y)
$$

and applies rotations:

$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

This is equivalent to multiplying a complex number:

$$
(x+iy)e^{i\theta}
$$

However, RoPE hides the complex structure by implementing it as paired real dimensions.

CoPE keeps the representation explicitly complex, preserving the algebraic operations of complex space.

---

## Positional Composition

RoPE primarily exposes relative position through attention:

$$
e^{ia\omega}\overline{e^{ib\omega}}
=
e^{i(a-b)\omega}
$$

The relative offset appears during the query-key interaction.

CoPE instead treats positional representations themselves as manipulable.

A model could explicitly compute:

$$
\mathrm{CoPE}(a)\odot\mathrm{CoPE}(b)
$$

to generate a representation of $a+b$.

Likewise:

$$
\mathrm{CoPE}(a)\odot\overline{\mathrm{CoPE}(b)}
$$

generates a representation of $a-b$.

This allows relative quantities to be dynamically constructed rather than only implicitly recovered through attention.

---

## Hypothesized Benefits

If positional information is represented as a composable algebraic object, neural networks may be able to perform abstract arithmetic directly in embedding space.

Potential examples:
- **Position:** $10+5=15$
- **Distance:** endpoint minus start point
- **Time:** current time plus duration
- **Rotation:** orientation plus rotation offset
- **Hierarchy:** parent level plus depth offset

The model no longer needs to learn these relationships entirely from examples. The representation itself provides the operation.

See [CoPE Hypothesis](cope-hypothesis.md) for formal experimental hypotheses and proposed experiments.

---

## Human note: GLU replacement for CoPE

Instead of any GLU, just do three projections A, B, and C from the embedding.
Then pass forwards `A * B + C` 
This gives a residual connection for deep stability, as well as a nice mix of potential operations
If a dimension in `(A * B)` equals -C, then it performs a gating operation on C
If a dimension in C is always 0, then it's purely an abstract addition on the encoded data via `A * B`

---

## Relation to Associative Memory

Attention can be viewed as a differentiable associative array:
- Keys act as addresses.
- Queries act as lookup requests.
- Values act as stored content.

Standard attention computes soft retrieval:

$$
\mathrm{softmax}(QK^T)V
$$

CoPE extends this view by making addresses themselves algebraically manipulable. Instead of only asking "which stored value matches this address?", the model can compute "what address results from combining these addresses?" This turns positional information from a passive feature into an active computational structure.

See [Associative Memory View](associative-memory-view.md) for the full treatment of attention as an associative array.

---

## See Also

- [CoPE Hypothesis](cope-hypothesis.md) — Formal hypotheses and experiments
- [CoPE Frequencies](cope-frequencies.md) — Parameterized frequency variants
- [CoPE-SoftOR MLP](cope-softor-mlp.md) — CoPE composition inside the MLP via Soft OR gates
- [CoPE-Gated Attention](cope-gated-attention.md) — Complex-valued gated attention
- [Associative Memory View](associative-memory-view.md) — Attention as a differentiable associative array
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan and design decisions
