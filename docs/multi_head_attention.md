# Multi-Head Attention Explained

This note explains single-head attention and multi-head attention from scratch.

It answers these questions:

- what are Q, K, V
- how they are created
- where the weight matrices come from
- how single-head attention works step by step
- how multi-head attention extends it
- what happens during training
- why this design is used

### The Starting Point

Suppose we have a sentence:

```text
"the cat sat"
```

The model does not see words. It sees token IDs.

After tokenization:

```text
[11, 502, 77]
```

### Step 1. Token Embeddings

Each token ID is looked up in an embedding table.

Let us pretend each token becomes a vector of size 3:

```text
"the" = [0.1, 0.2, 0.3]
"cat" = [0.4, 0.5, 0.6]
"sat" = [0.7, 0.8, 0.9]
```

We stack them into one matrix:

```text
x = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9]
]
```

Shape: (3 tokens, 3 dimensions)

### Step 2. Add Positional Information

Now we inject position so the model knows order.

```text
x = x + position_encoding
```

Now each vector carries both meaning and position.

### Step 3. Where Do Wq, Wk, Wv Come From

This is the most common confusion point.

Wq, Wk, Wv are **learned weight matrices**.

They are created when the model is initialized.

```python
Wq = nn.Linear(d_model, d_model, bias=False)
Wk = nn.Linear(d_model, d_model, bias=False)
Wv = nn.Linear(d_model, d_model, bias=False)
```

At initialization:

- they contain random small numbers
- they are not precomputed
- they are not derived from data
- they start as random parameters

During training:

- the optimizer updates them
- they slowly become meaningful
- the model learns what to look for and what to offer

So Wq, Wk, Wv are just trainable parameters like any other weights in a neural network.

### Step 4. Create Q, K, V

Now we project the input x through these three weight matrices.

```text
Q = x @ Wq
K = x @ Wk
V = x @ Wv
```

This means:

- Q is x transformed by Wq
- K is x transformed by Wk
- V is x transformed by Wv

All three start from the same x.

They just take different learned paths.

### Step 5. What Q, K, V Mean

Q stands for query.
K stands for key.
V stands for value.

Think of it like a database search.

Q asks: what am I looking for?
K says: what do I contain?
V says: what information do I pass forward?

For each token, the model learns:

- what to look for in other tokens (Q)
- what this token can offer to others (K)
- what this token actually contributes when selected (V)

### Step 6. Compute Attention Scores

Now we compare every Q with every K.

```text
scores = Q @ K.transpose()
```

This gives a matrix where each cell measures how much one token should attend to another.

Example scores matrix:

```text
       the   cat   sat
the  [ 5.1   2.3   0.1 ]
cat  [ 4.0   6.7   1.2 ]
sat  [ 1.1   3.4   7.8 ]
```

### Step 7. Scale The Scores

We divide by the square root of the head dimension.

```text
scores = scores / sqrt(d_head)
```

This keeps the values stable and prevents softmax from becoming too sharp.

### Step 8. Apply Causal Mask (Decoder-Only)

In a decoder-only model, a token cannot look at future tokens.

We add a mask with negative infinity to block future positions.

```text
scores = scores + causal_mask
```

After masking, future positions become very negative.

### Step 9. Softmax To Get Weights

We convert scores to probabilities.

```text
weights = softmax(scores, dim=-1)
```

Now each row sums to 1.

Example:

```text
       the   cat   sat
the  [ 0.90  0.09  0.01 ]
cat  [ 0.15  0.80  0.05 ]
sat  [ 0.02  0.10  0.88 ]
```

This means:

- "the" attends mostly to itself
- "cat" attends mostly to itself, a bit to "the"
- "sat" attends mostly to itself, some to "cat"

### Step 10. Mix Values Using Weights

Now we use these weights to combine V vectors.

```text
output = weights @ V
```

So if "sat" attends 88 percent to itself and 10 percent to "cat", its output will be mostly its own V with a small contribution from "cat"s V.

### Step 11. Output Projection

The attention output is projected one more time.

```text
final_output = output @ Wo
```

This mixes information from all heads later.

### Single-Head Attention Summary

The full single-head flow is:

```text
x = embeddings + position
Q = x @ Wq
K = x @ Wk
V = x @ Wv
scores = Q @ K.transpose()
scores = scores / sqrt(d_head)
scores = scores + mask
weights = softmax(scores)
output = weights @ V
final = output @ Wo
```

That is single-head attention.

### Why Multi-Head Attention

One attention head learns one pattern of relationships.

But language has many kinds of relationships.

Examples:

- subject-verb agreement
- pronoun references
- long-distance dependencies
- nearby word interactions
- code structure patterns

One head cannot capture all of these well.

So we use multiple heads in parallel.

### How Multi-Head Attention Works

Instead of one set of Q, K, V, we split into several heads.

If d_model is 128 and num_heads is 4, then each head works with dimension 32.

```text
head_dim = d_model / num_heads
```

For each head:

```text
Q_head = x @ Wq_head
K_head = x @ Wk_head
V_head = x @ Wv_head
head_output = attention(Q_head, K_head, V_head)
```

Then we concatenate all head outputs:

```text
concat = [head_1_output, head_2_output, head_3_output, head_4_output]
```

And project once more:

```text
final_output = concat @ Wo
```

### Multi-Head Attention Summary

The full multi-head flow is:

```text
x = embeddings + position

for each head:
    split x into head_dim chunks
    Q_h = x_h @ Wq_h
    K_h = x_h @ Wk_h
    V_h = x_h @ Wv_h
    scores_h = Q_h @ K_h.transpose()
    scores_h = scores_h / sqrt(head_dim)
    scores_h = scores_h + causal_mask
    weights_h = softmax(scores_h)
    out_h = weights_h @ V_h

concat all out_h
final = concat @ Wo
```

### Where Do The Weights Come From During Training

At the start:

- Wq, Wk, Wv, Wo are randomly initialized
- they contain small random numbers
- they do not mean anything yet

During the first forward pass:

- attention is computed using random weights
- the output is essentially noise
- the loss is very high

During backpropagation:

- gradients flow backward from the loss
- gradients tell each weight how to change
- Wq, Wk, Wv, Wo are updated slightly

After many steps:

- Wq learns what to look for
- Wk learns what to offer
- Wv learns what to contribute
- Wo learns how to combine head outputs

This is exactly how any neural network learns.

### Why Split Into Heads

If we did not split:

- one giant attention would try to learn everything
- it would be harder to specialize
- it would be less expressive

With multiple heads:

- each head can specialize
- some heads learn local patterns
- some heads learn long-range patterns
- some heads learn syntax
- some heads learn semantics

### Decoder-Only Specific Note

In a decoder-only model:

- Q, K, V all come from the same sequence
- this is self-attention
- the causal mask prevents looking at future tokens
- this makes generation left to right

### Final Mental Model

Keep this picture in your head:

```text
tokens become embeddings
embeddings get position
embeddings project to Q, K, V using learned weights
Q and K compute who should listen to whom
softmax turns scores into attention weights
weights mix V vectors
multiple heads do this in parallel
concatenate and project to get final output
```

That is multi-head self-attention.