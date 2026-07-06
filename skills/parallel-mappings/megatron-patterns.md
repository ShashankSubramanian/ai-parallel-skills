# Megatron-Style Patterns

These patterns describe where collectives usually appear. Adapt names and APIs
to the target framework.

## Column-Parallel Linear

Shard the output feature dimension of a linear layer.

```text
x: b l e
w_i: e f/tp
y_i: b l e, e f/tp -> b l f/tp
```

- Forward: each rank computes local output features.
- Backward: input gradient requires all-reduce across tensor-parallel ranks if
  the next layer consumed gathered output or if logical input gradients must be
  materialized.

## Row-Parallel Linear

Shard the input feature dimension of a linear layer.

```text
x_i: b l e/tp
w_i: e/tp f
y_partial_i: b l e/tp, e/tp f -> b l f
y = all_reduce(sum, y_partial_i)
```

- Forward: all-reduce partial outputs unless the next region can consume reduced
  shards.
- Backward: weight gradients are local to each input shard; input gradients stay
  sharded unless gathered for an upstream consumer.

## Attention Head Parallelism

Shard attention heads across tensor-parallel ranks.

```text
q_i, k_i, v_i: b l a/tp e_head
```

- Head-local attention usually does not need communication inside attention.
- Output projection often uses row-parallel linear and requires a reduction.
- Validate that number of heads is divisible by tensor-parallel size.

## Vocabulary Parallelism

Shard embedding or output vocabulary dimension.

- Embedding: each rank owns a vocabulary range; combine partial embeddings with
  masking plus reduction or gather.
- Output head: each rank computes logits for a vocabulary shard; loss may need
  parallel cross entropy or all-gathered logits.

## Placement Rule

Pair column-parallel and row-parallel layers to avoid unnecessary gathers:

```text
full input -> column-parallel -> sharded hidden -> row-parallel -> full output
```

Only gather when a non-parallel consumer requires the full logical tensor.
