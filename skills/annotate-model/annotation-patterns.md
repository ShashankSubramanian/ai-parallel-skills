# Annotation Patterns

Use this subskill to map common model modules to `comm_metadata`. Choose the
actual group names from the approved plan. Below are some common patterns.

## Tensor-Parallel Linear

Column-parallel weight:

```python
weight.comm_metadata = {
    "sharded": ["tp", None],
    "shared": ["sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
bias.comm_metadata = {
    "sharded": ["tp"],
    "shared": ["sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

Row-parallel weight:

```python
weight.comm_metadata = {
    "sharded": [None, "tp"],
    "shared": ["sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

If the row-parallel bias is added after the tensor-parallel output reduction, it
is usually unsharded but replicated across the model-parallel group:

```python
bias.comm_metadata = {
    "sharded": [None],
    "shared": ["tp-sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

## Norm Parameters

Norm affine parameters are usually full local tensors replicated across
model-parallel groups. If activations are spatially or context partitioned, their
weight gradients often need extra reduction:

```python
norm.weight.comm_metadata = {
    "sharded": [None],
    "shared": ["tp-sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

For head-local QK norm or other parameters whose gradients are local to a tensor
parallel shard but shared over all model-parallel axes, use the group required by
the plan, for example `tp-sp1-sp2`.

## Convolution And Patch Embedding

If the convolution weights are not partitioned but activations are spatially
partitioned:

```python
conv.weight.comm_metadata = {
    "sharded": [None, None, None, None],
    "shared": ["tp-sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

Use the same `shared` and `reduce` groups for bias unless the plan says
otherwise. An unsharded one-dimensional bias uses `"sharded": [None]`; if the
plan shards output channels, name that group instead.

## Positional Or Domain Parameters

Parameters that are already partitioned over spatial dimensions should mark
those tensor dimensions explicitly:

```python
pos_embed.comm_metadata = {
    "sharded": [None, None, "sp1", "sp2", None],
    "shared": ["tp"],
    "reduce": [],
}
```

## Wrapper Layers

If adding or using wrapper layers, accept a `comm_metadata`
argument and attach it to the created parameters inside the wrapper.

Do not hide ownership decisions inside wrappers. The caller should pass metadata
derived from the approved plan.
