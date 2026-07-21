# Benchmark Survey: Evaluating CoPE vs RoPE

## Summary

Survey of benchmarks usable for comparing Complex Positional Encoding (CoPE) against Rotary Positional Embedding (RoPE) and other positional encoding methods in transformer language models. Benchmarks are organized into four tiers by compute cost, from single-GPU minutes to multi-GPU weeks. Each entry includes the task, metric, dataset size, compute estimate, and rationale.

**Key hypothesis:** CoPE provides sample-efficiency and extrapolation advantages on tasks whose underlying structure is a group operation (addition, subtraction, relative distance, transformations) because composition is available as a first-class primitive inside the modified MLP, rather than something the network must discover entirely through learned parameters.

---

## Tier 0: Synthetic / Toy (single GPU, minutes to hours)

Most critical for isolating CoPE's advantage — directly test compositional arithmetic without language confounds.

### 0.1 Modular Arithmetic (Grokking)
- **Source:** Power et al. 2022 (`2201.02177`), Nanda et al. 2023 (`2301.05217`)
- **Task:** Predict `a + b mod p` from tokenized operands
- **Metric:** Accuracy (exact match)
- **Size:** ~p² examples; train on 30-50% split
- **Compute:** Minutes (p=59) to hours (p=113), single GPU
- **Relevance:** Nanda et al. showed the network solves this by learning a discrete Fourier transform circuit. CoPE provides complex exponentials natively — the model may grok faster or with less data. Test held-out moduli for extrapolation.

### 0.2 Position Arithmetic (Custom Synthetic)
- **Source:** Proposed in hypothesis discussion
- **Task:** Given CoPE-encoded positions, predict composed position. Input: `CoPE(a), CoPE(b)` → Output: `CoPE(a+b)` or `|a-b|`
- **Metric:** MSE on complex encoding or decoded position accuracy
- **Size:** Arbitrary (OOD splits at larger positions)
- **Compute:** Minutes, single GPU
- **Relevance:** Direct test of `CoPE(a)⊙CoPE(b) = CoPE(a+b)`. If the MLP provides this operation, needs orders of magnitude fewer examples.

### 0.3 Associative Recall
- **Source:** Ba et al. 2016; Olsson et al. 2022 (induction heads)
- **Task:** Given key-value pairs and a query key, retrieve associated value
- **Metric:** Accuracy
- **Size:** Variable (100s-1000s of pairs)
- **Compute:** Minutes, single GPU
- **Relevance:** Tests whether position-addressable retrieval benefits when address arithmetic is available.

### 0.4 Selective Copy
- **Source:** Mechanistic interpretability literature
- **Task:** Copy marked tokens from sequence, ignoring unmarked ones
- **Metric:** Exact match accuracy
- **Compute:** Minutes, single GPU
- **Relevance:** Tests position-based routing.

### 0.5 SCAN (Compositional Generalization)
- **Source:** Lake & Baroni 2018
- **Task:** Translate navigation commands to action sequences
- **Metric:** Exact sequence match accuracy
- **Format:** MCD splits (primitive, new combination, length)
- **Size:** ~21K examples
- **Compute:** <2 hours, single GPU
- **Relevance:** "Add primitive" and "length" splits test combining constituents at arbitrary positions.

### 0.6 COGS (Compositional Semantic Parsing)
- **Source:** Kim & Linzen 2020
- **Task:** Map sentences to logical forms; 14 generalization types
- **Metric:** Exact match accuracy
- **Size:** ~24K training, ~1K test
- **Compute:** <2 hours, single GPU
- **Relevance:** Stricter compositionality test than SCAN. Each generalization type tests a different compositional operation.

### 0.7 bAbI (Question Answering)
- **Source:** Weston et al. 2016
- **Task:** 20 synthetic QA tasks (counting, lists, paths, deduction, coreference)
- **Metric:** Per-task accuracy, mean across 20 tasks
- **Size:** 10K examples per task
- **Compute:** <1 hour, single GPU
- **Relevance:** Tasks 1-3 (fact retrieval), 4-5 (two-fact reasoning), 6 (counting), 7-9 (lists/sets), 11-14 (coreference) all involve position-dependent reasoning.

### 0.8 Reverse / Duplicate String
- **Source:** Common sanity check
- **Task:** Output reverse or duplicate of input sequence
- **Metric:** Exact match
- **Compute:** Minutes, single GPU
- **Relevance:** Length extrapolation — a model with address arithmetic should reverse arbitrary-length sequences more easily.

### 0.9 Parity / Bitwise XOR
- **Source:** Common algorithmic benchmark
- **Task:** Predict parity (sum mod 2) of bit sequence
- **Metric:** Accuracy
- **Compute:** Minutes, single GPU
- **Relevance:** Cumulative state tracking analogous to CoPE position composition.

---

## Tier 1: Small-Scale Language (single GPU, hours)

### 1.1 LAMBADA
- **Source:** Paperno et al. 2016
- **Task:** Predict last word of passage requiring reading comprehension
- **Metric:** Accuracy
- **Size:** ~10K test
- **Compute:** Hours (~100M param model), single GPU
- **Relevance:** Discourse position tracking and long-range dependencies.

### 1.2 BLiMP
- **Source:** Warstadt et al. 2020
- **Task:** Grammatical acceptability (67 subtasks: anaphor agreement, argument structure, binding, control, NPI licensing, island effects)
- **Metric:** Per-subtask accuracy, mean
- **Size:** ~10K sentences total
- **Compute:** Few hours, single GPU
- **Relevance:** Many subtests involve position-sensitive phenomena.

### 1.3 PIQA
- **Source:** Bisk et al. 2020
- **Task:** Physical commonsense reasoning
- **Metric:** Accuracy
- **Size:** ~16K training, ~3K test
- **Compute:** Hours, single GPU
- **Relevance:** Physical reasoning involves relative position, ordering, action sequences.

### 1.4 HellaSwag
- **Source:** Zellers et al. 2019
- **Task:** Commonsense sentence completion
- **Metric:** Accuracy
- **Size:** ~40K training, ~10K test
- **Compute:** Hours, single GPU
- **Relevance:** Narrative position and temporal ordering.

### 1.5 WinoGrande
- **Source:** Sakaguchi et al. 2021
- **Task:** Pronoun resolution (Winograd schema)
- **Metric:** Accuracy
- **Size:** ~44K training
- **Compute:** Hours, single GPU
- **Relevance:** Coreference requires tracking position and structural relationships.

### 1.6 GSM8K
- **Source:** Cobbe et al. 2021
- **Task:** Grade-school math word problems (multi-step arithmetic)
- **Metric:** Accuracy (with or without chain-of-thought)
- **Size:** ~7.5K training, ~1.3K test
- **Compute:** Hours, single GPU
- **Relevance:** Multi-step arithmetic — CoPE's additive algebra could help maintain numerical state across steps.

### 1.7 MathQA
- **Source:** Amini et al. 2019
- **Task:** Math word problems with annotated operations
- **Metric:** Accuracy
- **Size:** ~30K training
- **Compute:** Hours, single GPU
- **Relevance:** Similar to GSM8K; operation annotations enable deeper analysis.

### 1.8 SuperGLUE
- **Source:** Wang et al. 2019
- **Task:** 8 NLU tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC)
- **Metric:** Per-task accuracy/F1, mean
- **Size:** ~10-100K per task
- **Compute:** ~1-2 days for all tasks, single GPU
- **Relevance:** COPA (causal reasoning about event ordering) and ReCoRD (long-range retrieval) are most likely to show position-encoding differences.

### 1.9 ARC (AI2 Reasoning Challenge)
- **Source:** Clark et al. 2018
- **Task:** Science multiple-choice (Easy and Challenge sets)
- **Metric:** Accuracy
- **Size:** ~3K Easy, ~2.5K Challenge
- **Compute:** Hours, single GPU
- **Relevance:** Challenge set requires multi-step inference.

### 1.10 WikiText-103
- **Source:** Merity et al. 2017
- **Task:** Word-level language modeling perplexity
- **Metric:** Perplexity
- **Size:** ~100M tokens
- **Compute:** Hours to 1 day, single GPU
- **Relevance:** Used in ALiBi paper. Standard LM benchmark.

---

## Tier 2: Mid-Scale (multiple GPUs, 1-3 days)

### 2.1 MMLU
- **Source:** Hendrycks et al. 2021
- **Task:** 57 subjects across STEM, humanities, social sciences (multiple choice)
- **Metric:** Accuracy per subject, mean
- **Size:** ~14K test
- **Compute:** 1-4 GPUs, 1-3 days fine-tuning
- **Relevance:** Broad knowledge — useful as control (CoPE should not degrade general capabilities).

### 2.2 BIG-Bench Hard (BBH)
- **Source:** Suzgun et al. 2022 (subset of Srivastava et al. 2022)
- **Task:** 23 challenging reasoning tasks (logic, math, algorithm, tracking, etc.)
- **Metric:** Accuracy per task
- **Size:** Few-shot (3-5 examples per task)
- **Compute:** 1-4 GPUs, 1-2 days
- **Relevance:** "Tracking_shuffled_objects", "navigate", "dyck_languages", "geometric_shapes" directly involve position manipulation.

### 2.3 LRA (Long Range Arena)
- **Source:** Tay et al. 2021
- **Task:** 6 long-context tasks (ListOps, Text, Retrieval, Image, Pathfinder, Path-X)
- **Metric:** Accuracy
- **Size:** Varies by subtask
- **Compute:** 1 GPU, hours to 1 day
- **Relevance:** Position encoding quality directly impacts long-range attention performance.

### 2.4 LongBench
- **Source:** Bai et al. 2023
- **Task:** 6 categories of long-context tasks (single-doc QA, multi-doc QA, summarization, few-shot, synthetic, code)
- **Metric:** Per-task (F1, Rouge-L, accuracy)
- **Size:** ~4.8K examples across 21 tasks
- **Compute:** 1-4 GPUs, 1-2 days
- **Relevance:** Long-sequence position coherence.

### 2.5 SCROLLS
- **Source:** Shaham et al. 2022
- **Task:** Long-document QA, summarization, NLI
- **Metric:** Task-specific (Rouge, F1, accuracy)
- **Size:** 7 long-document datasets
- **Compute:** 1-4 GPUs, 1-3 days
- **Relevance:** Complementary long-document coverage to LongBench.

### 2.6 RULER
- **Source:** Hsieh et al. 2024
- **Task:** 6 categories of long-context synthetic retrieval (single-needle, multi-needle, multi-value, multi-query, temporal, aggregation)
- **Metric:** Accuracy
- **Size:** Synthetic, arbitrary
- **Compute:** 1 GPU, hours
- **Relevance:** Temporal retrieval subtask directly tests position arithmetic.

### 2.7 SWE-bench
- **Source:** Jimenez et al. 2024
- **Task:** Resolve real GitHub issues by editing code
- **Metric:** Resolution rate (% correctly resolved)
- **Size:** ~2K task instances
- **Compute:** 1-4 GPUs, days
- **Relevance:** Complex multi-step reasoning. High-signal control.

---

## Tier 3: Large-Scale (multiple GPUs, days to weeks)

### 3.1 The Pile Perplexity
- **Source:** Gao et al. 2020
- **Task:** Language modeling perplexity on diverse corpus
- **Metric:** Perplexity
- **Size:** ~800 GB, ~350B tokens
- **Compute:** 4-8 GPUs, weeks
- **Relevance:** Authoritative LM benchmark. CoPE advantage expected on compositional tasks rather than raw perplexity.

### 3.2 Full Pre-train + Fine-tune
- **Source:** Standard paradigm
- **Task:** Pre-train on C4 or The Pile, fine-tune on GLUE/SuperGLUE
- **Metric:** Downstream task scores
- **Compute:** 4-8 GPUs, weeks
- **Relevance:** Ultimate validation.

### 3.3 HELM
- **Source:** Liang et al. 2022
- **Task:** ~40 scenarios across 7 categories
- **Metric:** Multiple per-scenario metrics
- **Compute:** 4-8 GPUs, weeks
- **Relevance:** Most comprehensive evaluation; use only after CoPE shows promise on smaller benchmarks.

### 3.4 Needle-in-a-Haystack (NIAH)
- **Source:** Long-context stress test; formalized in Arora et al. 2024 and RULER paper
- **Task:** Retrieve specific fact buried in long distractor text
- **Metric:** Retrieval accuracy vs. context length and needle position
- **Size:** Synthetic, arbitrary length
- **Compute:** 1 GPU, hours (inference only — no training)
- **Relevance:** Tests position-addressable retrieval at arbitrary positions.

---

## Tier 4: Mechanistic Interpretability Probes

### 4.1 Position Probe
- **Source:** Haviv et al. 2022 methodology
- **Task:** Train linear probe on hidden states to predict absolute/relative position
- **Metric:** Probe accuracy
- **Compute:** Single GPU, minutes
- **Relevance:** Tests whether CoPE produces cleaner positional representations in hidden states.

### 4.2 Activation Patching
- **Source:** Marks & Tegmark 2024 methodology
- **Task:** Intervene on residual stream directions and measure effect on position-sensitive outputs
- **Metric:** Causal effect size
- **Compute:** Single GPU, hours
- **Relevance:** Tests whether the network actually uses CoPE algebra for computation.

### 4.3 Attention Head Fourier Analysis
- **Source:** Nanda et al. 2023 (progress measures for grokking)
- **Task:** Analyze Fourier content of attention patterns; identify phase-based circuits
- **Metric:** Fourier coefficient magnitudes
- **Compute:** Single GPU, hours
- **Relevance:** CoPE provides complex exponentials natively — heads may show more structured Fourier patterns from initialization.

---

## Recommended Test Sequence

### Phase 0: Sanity checks (minutes)
- 0.2 Position Arithmetic
- 0.4 Selective Copy
- 0.8 Reverse String

### Phase 1: Synthetic composition tests (hours, 1 GPU)
- 0.1 Grokking modular arithmetic (p=59, 83, 113)
- 0.3 Associative Recall
- 0.9 Parity / XOR

### Phase 2: Compositional generalization (hours, 1 GPU)
- 0.5 SCAN
- 0.6 COGS

### Phase 3: Language understanding (1-2 days, 1 GPU)
- 1.1 LAMBADA
- 1.2 BLiMP
- 1.3 PIQA, 1.4 HellaSwag, 1.5 WinoGrande
- 1.6 GSM8K, 1.7 MathQA
- 1.8 SuperGLUE

### Phase 4: Long-context (1-2 days, 1-4 GPUs)
- 2.3 LRA
- 2.4 LongBench
- 2.6 RULER
- 3.4 NIAH

### Phase 5: Large-scale validation (days-weeks, 4-8 GPUs)
- 2.1 MMLU
- 2.2 BBH
- 3.1 Pile perplexity
- 3.2 Full pre-train + fine-tune

### Phase 6: Mechanistic analysis (hours, 1 GPU)
- 4.1 Position probe
- 4.2 Activation patching
- 4.3 Fourier/circuit analysis

---

## Hardware & Software Requirements

### DGX Spark (128 GB unified memory)
Capable of training models up to ~1-2B parameters with appropriate batch sizes.
- Tier 0-1: Easily fits
- Tier 2: Most tasks fit with batch size tuning
- Tier 3: Pre-training beyond ~1B is challenging but feasible for smaller checkpoint runs

### Software stack (existing in repo)
- PyTorch 2.13 + torch (already installed)
- Need: `transformers`, `datasets`, `accelerate`, `wandb` or `tensorboard`, `einops`
- For RoPE baseline: use Hugging Face transformers implementation
- For CoPE: implement custom complex position module

### Evaluation frameworks
- **lm-evaluation-harness** (EleutherAI): standard API for most LM benchmarks
- **longbench** (THUDM): dedicated long-context evaluation
- **RULER** (Hsieh et al.): synthetic long-context retrieval
- **HELM** (Stanford CRFM): comprehensive evaluation suite
- **grok** (OpenAI): modular arithmetic dataset generation

---

## Baseline Models to Compare

For each benchmark, compare these configurations (all else equal):

1. **No Positional Encoding** (NoPE) — baseline from Haviv et al. 2022
2. **Learned Absolute Embeddings** — standard pre-RoPE approach
3. **RoPE (Rotary)** — current dominant approach, primary baseline
4. **ALiBi** — strong length-generalization baseline
5. **CoPE (Complex Positional Encoding)** — the proposed method

Use identical architecture (depth, width, heads, training hyperparameters) differing only in the positional encoding method.

---

## Key Controls

1. **Model size.** CoPE changes representational capacity. Control by comparing models with matched total parameters, and also matched hidden dimensions.
2. **Compute-matched curves.** Report loss vs. step, not just final metrics.
3. **Permutation baseline (Hypothesis Experiment 3).** Include a task where position is randomly mapped to labels — CoPE should NOT help. This rules out "CoPE is just a stronger architecture" as the explanation.
4. **OOD position extrapolation.** Always test on positions/lengths not seen during training. This is where CoPE's algebraic composition is predicted to matter most.
5. **No cherry-picking.** Report all benchmarks attempted, not only favorable ones.

---

## Design Decisions

### Decision 1: How CoPE(pos) combines with the token embedding — ✅ RESOLVED

**Chosen: Elementwise multiply.** $h_i = \text{embed(token}_i) \odot \text{CoPE}(i)$

Matching the associative-array interpretation where positional information is an address that modulates content. See [Complex Positional Encoding (CoPE)](cope.md) for full details on embedding integration.

### Decision 2: Modified MLP architecture — ✅ RESOLVED

**Chosen: Softmax-based MAX (OR) replaces SwiGLU gating.**

The MLP structure:

```
A = W_A(h)     # first projection → d_hidden
B = W_B(h)     # second projection → d_hidden

# For each pair (A_i, B_i):
#   out_i = sum([A_i, B_i] · softmax([A_i, B_i]))
#         = max(A_i, B_i)  (differentiable approximation of ternary OR)

out_i = softmax_max_pair(A_i, B_i)   # per-pair, d_hidden outputs

h_out = W_down(out)                  # project back to d_model
```

Softmax is applied over each 2-element pair independently, acting as a differentiable argmax. At $T=0$ this is exactly $\max(A_i, B_i)$, which is the ternary OR gate over $\{-1,0,+1\}$. AND emerges via negation through learned weights (AND = $-\max(-A,-B) = \min(A,B)$ via De Morgan). Learned linear projections can absorb sign flips, so the network can construct any binary logic gate for each hidden dimension.

Replaces SwiGLU's $\text{SiLU}(W_{\text{gate}} h) \odot (W_{\text{up}} h)$.

See [Soft OR Gates](soft-or-gates.md) for the full theory and [CoPE-SoftOR MLP](cope-softor-mlp.md) for the CoPE composition integration.

### Decision 3: Frequencies — ✅ RESOLVED

**Chosen: Fixed geometric series matching RoPE** ($\text{base}=10000$, $\omega_i = \text{base}^{-2i/d}$) for the initial comparison. Parameterized frequencies ($\omega_j = e^{\theta_j}$ with learned $\theta_j$) are deferred as a follow-up experiment.

See [CoPE Frequencies](cope-frequencies.md) for the parameterized frequency formulation.

### Decision 4: Model architecture matrix (Z-shaped comparison) — ⏳ IN PROGRESS

**Key principle:** Complex weight matrices are required for CoPE because RoPE hard-codes rotations into its linear transformations via 2×2 coupled blocks. CoPE without complex matrices would lack this hard-coded offset capability and be at a severe disadvantage. See [Complex Positional Encoding (CoPE)](cope.md) for details.

For a reference dimension $D$ per tier, four configurations form a Z-shaped comparison:

| Config | Type | dims | rotational units | real params (rough) |
|--------|------|------|------------------|-------------------|
| 1 | RoPE | $D$ | $D/2$ | $\sim D^2$ |
| 2.A | CoPE | $D/\sqrt{2}$ | $D/\sqrt{2}$ | $\sim D^2$ |
| 2.B | RoPE | $D/\sqrt{2}$ | $D/(2\sqrt{2})$ | $\sim D^2/2$ |
| 3 | CoPE | $D/2$ | $D/2$ | $\sim D^2/2$ |

Parameter matching: CoPE's complex matrices have 2 real parameters per element, so $2 \cdot (D/\sqrt{2})^2 = D^2$ for config 2.A (matching RoPE config 1), and $2 \cdot (D/2)^2 = D^2/2$ for config 3 (matching RoPE config 2.B). At matched parameter budgets, CoPE always has $\sqrt{2} \times$ more rotational units than RoPE.

**Pending:** Exact $D$ values per tier, handling of division constraints for multi-head attention.

### Decision 5: Training hyperparameters — ⏳ IN PROGRESS

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [CoPE Hypothesis](cope-hypothesis.md) — Experimental hypotheses
- [CoPE Frequencies](cope-frequencies.md) — Parameterized frequency variants
- [Gated Attention](gated-attention.md) — Gated attention mechanism
- [Soft OR Gates](soft-or-gates.md) — Ternary logic in MLPs
- [CoPE-SoftOR MLP](cope-softor-mlp.md) — CoPE composition via Soft OR gates
- [CoPE-Gated Attention](cope-gated-attention.md) — Complex-valued gated attention
- [Associative Memory View](associative-memory-view.md) — Attention as associative array
