# Parallelism Contract

The parallelism contract refers to a string representation of the sharded or local tensor shapes involved or mathematical operations. This references formalizes how to write tensor and operation contracts for sharding any layer and includes costs associated with the sharding.
If not specificied, use the following conventions:
- `dp`: data parallelism.
- `tp`: tensor parallelism.
- `cp`: context parallelism.
- `sp`: spatial parallelism.
- `seqp`: sequence parallelism.


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

For matmuls, if the inner / contracting /common dimension is sharded, then the operation computes a partial sum and requires a reduce collective (typically all-reduce) to sync the result. For example:

```text
b l e/tp, e/tp f -> b l f
```

`b l f` is a partial sum and needs to be all-reduced since the output is replicated across the `tp` ranks. If the output is sharded, then the collective is a reduce-scatter, for example, to `b l/tp f`.

If the inner / contracting dimension is not sharded, then the operation computes the full result and does not require a reduce collective. For example:

```text
b l e, e f/tp -> b l f/tp
```

Make sure local matmuls are legal and produce legal sharded tensors. If not, the sharded operation is either invalid or needs other collectives to ensure correctness. For example,

```text
b l/cp e, e l'/cp -> b l/cp l'/cp
```

is illegal because the output is sharded twice across `cp` ranks, which is not allowed. The correct operation is:

```text
b l/cp e, e l' -> b l/cp l'
```

which requires the second tensor / operand to be all-gathered first across the `cp` ranks.

## Backward Duals

In training mode, the backward pass typically requires a the conjugate communication collectives of the forward operation, and when analyzing forward and backward matmuls, you can think of the same collective having two different roles in forward and backward, making it a differentiable operation. See `collectives.md` for the list of common differentiable collectives and their contracts.
