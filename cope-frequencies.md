# CoPE Parameterized Frequencies

## Parameterized Frequencies

The source document defines CoPE as:

$$
\mathrm{CoPE}(p)=e^{ip\omega}
$$

where $\omega$ is a vector of rotational frequencies. In the general case, each element of this complex vector represents one independent rotational unit — there are as many frequency components as dimensions, and the exponential is applied elementwise.

The standard approach (inherited from RoPE) fixes $\omega$ to a geometric series determined by a base hyperparameter:

$$
\omega_i = \text{base}^{-2i/d}
$$

This imposes a specific frequency structure before training begins. A natural generalization is to make $\omega$ learnable.

---

### Parameterization

To keep frequencies strictly positive and stabilize the optimization landscape, each frequency is parameterized via an exponentiated scalar:

$$
\omega_j = e^{\theta_j}
$$

where $\theta_j$ is a free learnable parameter for dimension $j$, and $\theta \in \mathbb{R}^d$ is the full parameter vector (with $d$ the model dimension). The CoPE encoding becomes:

$$
\mathrm{CoPE}(p) = e^{ip \odot e^{\theta}}
$$

with the inner exponential applied elementwise.

#### Gradient properties

$$
\frac{\partial}{\partial\theta_j} e^{ip\omega_j}
= ip \cdot \omega_j \cdot e^{ip\omega_j}
= ip \cdot e^{\theta_j} \cdot e^{ip \cdot e^{\theta_j}}
$$

The gradient at each frequency scales proportionally to the frequency itself. Low frequencies receive small updates; high frequencies receive larger updates. This creates a natural multiplicative dynamic during training.

---

### Base Is Obsolete

The traditional base hyperparameter (e.g. $\text{base} = 10000$) controls the spacing of the geometric frequency series. With parameterized frequencies, this construct is unnecessary:

- Any geometric series can be recovered by setting $\theta_j = -\frac{2j}{d}\log(\text{base})$.
- Changing the base is equivalent to adding a constant offset to all $\theta_j$ parameters.
- Making each frequency independently learnable subsumes any fixed-base or learnable-base scheme.

The base abstraction adds complexity without expressive benefit. In the parameterized formulation, it is simply a particular initialization choice for $\theta$.

---

### Layer-wise Frequencies

> **Status: Open question — further consideration needed.**

An open architectural question is whether frequencies should be:

- **Shared across layers**: A single $\theta$ vector is learned and applied identically at every transformer layer.
- **Layer-specific**: Each layer $l$ learns its own $\theta^{(l)}$.

**Arguments for shared frequencies:**
- Fewer parameters; simpler optimization.
- The algebraic structure (composition via multiplication) is layer-invariant — position means the same thing at every depth.
- Consistent with how RoPE applies the same rotation at every layer.

**Arguments for layer-specific frequencies:**
- Different layers may specialize in different representational scales (lower layers: local syntax, higher layers: long-range semantics).
- Allowing each layer to tune its frequency basis could improve multi-scale representation learning.
- Compatible with findings from HARoPE and other adaptive-frequency approaches.

**Arguments against layer-specific frequencies:**
- If compositionality is a core benefit of CoPE, then breaking the shared frequency basis across layers means the composition operation ($\odot$) no longer has a consistent algebraic interpretation across depths.
- A model could learn to use $\mathrm{CoPE}^{(l)}(a) \odot \mathrm{CoPE}^{(l)}(b)$ at layer $l$ to get $a+b$, but this representation would not be directly composable at layer $l+1$ if the frequencies differ.

Empirical comparison is needed to resolve this question.

---

### Initialization

The choice of initial $\theta$ determines the starting frequency distribution. Multiple strategies are worth considering.

#### Geometric (RoPE-matching)

Initialize $\theta$ to reproduce the standard RoPE geometric series:

$$
\theta_j = -\frac{2j}{d}\log(\text{base})
$$

This starts the model at the familiar RoPE frequency distribution and lets it drift during training. The base parameter serves only as an initial condition, not a fixed constraint.

Common choices:
- $\text{base} = 10000$ (standard RoPE)
- $\text{base} = 500000$ (used in longer-context models like LLaMA 3)

#### Uniform in log-space

Sample each $\theta_j$ uniformly in log-frequency space:

$$
\theta_j \sim \mathcal{U}(\log\omega_{\min}, \log\omega_{\max})
$$

This provides no prior on which frequencies are important, letting the model discover the structure entirely from data.

#### Learned initialization (warm-start)

Pre-train a small proxy model to estimate useful frequencies, then use the resulting $\theta$ as the initialization for the full model.

#### Head-specific initialization

For multi-head attention, different heads could begin with different frequency ranges, encouraging heads to specialize in different positional resolutions from the start.

---

### Relationship to Existing Work

Several recent approaches explore learnable or adaptive position encoding frequencies. The key distinctions are:

- **HARoPE** learns a linear transformation (via SVD) applied before the rotary mapping, with head-specific transformations. This learns a mixing of frequency components rather than the frequencies themselves.

- **Bifocal Attention** proposes "Spectral Evolution", where frequencies are initialized to the standard geometric series and allowed to evolve via gradient descent. This is the closest existing work to the parameterization described here. Bifocal Attention motivates learnable frequencies by identifying "spectral rigidity" in standard RoPE — the fixed geometric decay optimized for local syntax fails to capture long-range periodic structure needed for algorithmic reasoning.

The parameterized CoPE formulation unifies these ideas: $\theta$ is simply a vector of learnable parameters, the exponential enforces positivity, and the base and its geometric series become just one possible initialization.

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [CoPE Hypothesis](cope-hypothesis.md) — Experimental hypotheses
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan, including Design Decision 3 (frequency choice)
