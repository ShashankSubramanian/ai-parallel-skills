# Parallelism Contract

The parallelism contract is a string representation of the local tensor shapes in a mathematical operation. This reference defines how to write tensor and operation contracts, including the communication required by each sharding choice.
If not specified, use the following conventions:

- `dp`: data parallelism.
- `tp`: tensor parallelism.
- `cp`: context parallelism.
- `sp`: spatial parallelism.

Sequence parallelism is written as `sequence_parallel=true`. It is an extension
of tensor parallelism that shards eligible sequence activations with the
existing `tp` group. It is not a separate mesh dimension: a sequence-parallel
local shape may be written as `b l/tp e`.

## Device Mesh

Given total number of devices, `N`, a device mesh is a logical grouping of the `N` devices into a hyperrectangle. For example, `N = 16`, can be a device mesh of `4 x 4` or `2 x 8` or `2 x 2 x 4` or `2 x 4 x 1 x 2`, etc. Each dimension of the hyperrectangle / mesh forms a process group of devices that shard the logical tensor in some dimension and work together with communication collectives to ensure correctness.

## Tensor Contract for Sharded / Local Tensors

When planning parallelism and sharding, write the local tensor shapes keeping in mind the device mesh. For example, if `tp` devices are used to shard `l`, then the local tensor shape is:

```text
b l/tp e
```

Tensors can either be sharded or replicated across the device mesh. For example:
If the device mesh is `dp x tp x cp`, and if the tensor is `b/dp l/cp e`, then the tensor is sharded across `dp` and `cp`, but replicated across `tp`.

Illegal sharding is when the tensor layout needs more mesh dimensions than the device mesh provides. For example: if the device mesh is `dp x tp`, a tensor such as `b/dp l/tp e/tp` is illegal because a single `tp` mesh dimension cannot shard two independent axes at once. That layout would need separate mesh dimensions, such as `dp x tp_l x tp_e`.

## Model Operations Contract for Sharded Tensors

Model operations may or may not introduce communication collectives. See `collectives.md` for the list of collectives and their contracts.

For a two-input einsum, first write the logical contract:

```text
left labels, right labels -> output labels
```

- **Shared batch labels** appear in both inputs and the output.
- **Contracting labels** appear in both inputs but not the output.
- **Free labels** appear in exactly one input and the output. A model batch dimension is free when the other input is broadcast over it.

For example, in `b l e, b e f -> b l f`, `b` is a shared batch label. In `b l e, e f -> b l f`, `b` is a free label because the weight matrix is broadcast over the batch.

Then apply these sharding rules:

1. **A contracting dimension is sharded in both inputs.** Each rank computes a partial sum. All-reduce the result if it is replicated, or reduce-scatter it if the result should be sharded.

```text
b l e/tp, e/tp f -> b l f
# all-reduce b l f
```

1. **A contracting dimension is sharded in only one input.** The local contracting dimensions do not match. All-gather the sharded input before the einsum. Alternatively, shard the other input the same way and apply rule 1.

```text
b l e/tp, e f -> invalid
# all-gather the left input
b l e, e f -> b l f
```

1. **A free dimension is sharded.** The output inherits that shard. No collective is required. This includes a batch dimension that appears in only one input.

```text
b/dp l e, e f -> b/dp l f
```

2. **A batch dimension is sharded in both inputs.** If the shards are aligned, the output inherits the batch shard. No collective is required.

```text
b/dp l e, b/dp e f -> b/dp l f
```

3. **A shared batch dimension is sharded in only one input.** The label appears in both inputs, but their local sizes do not match. All-gather the sharded input, or shard the other input identically.

```text
b/tp l e, b e f -> invalid
# all-gather the left input
b l e, b e f -> b l f
```

4. **One mesh axis shards different free dimensions in the two inputs.** The output would use the same mesh axis twice, which is illegal. All-gather one input along that mesh axis.

```text
b l/cp e, e l'/cp -> b l/cp l'/cp  # illegal
# all-gather the right input
b l/cp e, e l' -> b l/cp l'
```

These rules apply per mesh axis. Apply them independently for multi-dimensional meshes, and verify that every local einsum has matching label sizes and produces a legal output layout.

## Backward Duals

In training mode, the backward pass typically requires the conjugate communication collectives of the forward operation. When analyzing forward and backward einsums, the same collective can have different roles in each direction, making it a differentiable operation. See `collectives.md` for the list of common differentiable collectives and their contracts.

For every trainable parameter, write its weight-gradient computation as an
einsum of the saved activation and output gradient, then apply the sharding
rules above to that wgrad einsum. A model-parallel group requires a wgrad
reduction when its ranks hold the same local parameter shard but each rank
computes a partial, non-duplicate gradient because the group shards a
contracting label. If the wgrad instead inherits the parameter's shard along
that group, the ranks hold different local gradients and must not reduce them
together. When several mesh dimensions jointly require one reduction, name
their combined group token.