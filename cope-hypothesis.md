# CoPE Hypothesis

The hypothesis I would extract from your argument is:

> **Hypothesis:** A transformer architecture that preserves positional information as an explicitly composable complex-valued representation will learn and generalize relational arithmetic (addition/subtraction of latent quantities, especially positions) more easily than a transformer where positional information is only injected as a modification to attention scores.

A weaker, easier-to-test version:

> **Hypothesis:** If a model has access to an operation where
> 
> $$
> C(a)\odot C(b)=C(a+b)
> $$
> 
> and
> 
> $$
> C(a)\odot \overline{C(b)}=C(a-b)
> $$
> 
> then it will learn relative-position tasks with fewer parameters, fewer examples, or better extrapolation than a model without that operation.

The key word is **learn**. The question is not whether the algebra is valid. It is whether exposing the algebraic structure gives optimization an advantage.

---

## Experiment 1: Pure positional arithmetic (smallest possible test)

Don't start with language. Start with synthetic tokens.

Create a dataset:

```
A at position 17
B at position 42

Question:
distance(A,B)?
```

The answer is:

```
25
```

Train three tiny transformers:

### Model A: learned absolute embeddings

Standard:

$$
x_i = token_i + position_i
$$

### Model B: RoPE

Standard rotary attention.

### Model C: CoPE

Give each token:

$$
x_i = token_i \odot CoPE(i)
$$

and add a learnable operation:

$$
z = x \odot W(x)
$$

or explicitly:

$$
z = CoPE(i)\odot \overline{CoPE(j)}
$$

for relational queries.

---

Test:

Train on positions:

```
0-100
```

Evaluate on:

```
100-1000
```

The question:

Can the model extrapolate?

A model that has learned "subtract positions" should generalize. A model that has memorized position relationships should fail.

---

## Experiment 2: Addition composition

This tests your "small + medium = big" idea.

Encode quantities:

```
token A = quantity 7
token B = quantity 13
```

Ask:

```
A + B = ?
```

But make the quantities represented only as CoPE phases:

$$
C(n)=e^{in\omega}
$$

Then the model receives:

$$
C(7),C(13)
$$

and must output:

$$
C(20)
$$

The important comparison:

### Baseline

Network must learn:

```
embedding(7) + embedding(13) → embedding(20)
```

### CoPE

Network can discover:

$$
C(7)\odot C(13)=C(20)
$$

If your hypothesis is right, the CoPE model should need dramatically fewer examples.

---

## Experiment 3: Remove the shortcut

This is the most important control.

A skeptic could say:

"Of course your model wins; you gave it multiplication."

So create two tasks:

### Task A: additive group

```
position + offset
```

CoPE should win.

### Task B: arbitrary permutation

```
position → random label
```

CoPE should not help.

If CoPE improves both, it is just a stronger architecture.  
If it specifically improves additive tasks, it is exploiting the intended structure.

---

## Experiment 4: Look inside the learned representations

Train a model normally.

Then extract hidden states:

$$
h_0,h_1,h_2,...
$$

Ask whether there exists a linear map (W) such that:

$$
W(h_a\odot h_b)\approx h_{a+b}
$$

or:

$$
W(h_a\odot\bar h_b)\approx h_{a-b}
$$

If the model spontaneously discovers this algebra, then your hypothesis is partially true even without explicit CoPE.

---

## Strongest prediction

The place I would expect your idea to shine is not ordinary sequence position.

Transformers already have enormous capacity to memorize relative positions.

The interesting target is **latent quantities**:
- elapsed time
- spatial transformations
- rotations
- hierarchy depth
- causal distance
- ordering relationships

Anything where the correct operation is naturally a group operation.

For example:

```
Paris
 + 3 hours
 =
time-zone-adjusted Paris time
```

or:

```
object pose
 + rotation
 =
new pose
```

Those are exactly the cases where "encode a quantity as a group element, then compose by multiplication" is the natural mathematical representation.

So the cleanest falsifiable claim I see is:

> Complex-valued compositional embeddings provide a measurable sample-efficiency and extrapolation advantage on tasks whose underlying structure is a group operation, because the model no longer has to learn the group law from data.

That is the experiment I would run first.

---

## Testable Hypothesis

A transformer with explicit complex compositional positional representations should learn tasks involving additive relationships more efficiently and extrapolate better than models where positional relationships must be inferred.

The strongest expected advantage is on tasks with an underlying group structure:
- addition
- subtraction
- relative distance
- transformations
- rotations

The key experimental question is whether exposing the algebraic structure improves learning compared with forcing the network to discover the same structure through ordinary parameters.

---

## See Also

- [Complex Positional Encoding (CoPE)](cope.md) — Core architecture
- [CoPE Frequencies](cope-frequencies.md) — Parameterized frequency variants
- [Benchmark Survey Plan](benchmark-survey-plan.md) — Evaluation plan with specific benchmark tiers
