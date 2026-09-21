# Metadata Contract

Use this subskill to define the `comm_metadata` schema for trainable
parameters.

## Schema

```python
comm_metadata = {
    "sharded": list[str | None],
    "shared": list[str],
    "reduce": list[str],
}
```

## Fields

- `sharded`: One entry per parameter tensor dimension. A string means that
  tensor dimension is partitioned across the named communication group. `None`
  means that tensor dimension is local/full on each rank. An unsharded
  rank-`n` tensor uses `[None] * n`; `[]` is valid only for a scalar parameter
  because its `ndim` is zero.
- `shared`: Communication groups whose ranks hold replicated copies of this
  parameter in addition to data parallelism. These groups must be synchronized
  after initialization.
- `reduce`: Model-parallel groups whose ranks hold the same local parameter
  shard but compute partial, non-duplicate gradients that must be all-reduced
  in addition to ordinary data parallelism.

## Logical Versus Stored Layout

Operation contracts use mathematical dimension order, while `sharded` always
uses the actual parameter tensor's dimension order. Record the translation in
the planning handoff before annotation. For example, a linear operation may use
logical weight `e f`, but `torch.nn.Linear.weight` is stored as
`(out_features=f, in_features=e)`. Sharding logical `f` therefore produces local
storage `(f/tp, e)` and metadata `["tp", None]`.

## DP Convention

Do not encode ordinary data parallelism in every parameter's metadata. The
selected DDP or FSDP wrapper owns ordinary communication over `dp`. Use
`reduce` only for extra model-parallel reductions such as `sp1-sp2`, `cp`, or
`tp-sp1-sp2`. FSDP's transient storage shards are not logical entries in
`sharded`.

## Shape Rules

- `len(sharded)` must equal `param.ndim`, including zero-dimensional scalar
  parameters.
- `None` means the corresponding stored parameter dimension is not sharded.
- A group token means the corresponding stored parameter dimension is
  partitioned across that group.
- The local dimension size must match the planned global dimension size divided
  by the group size, unless the approved plan specifies padding or uneven
  sharding.
- All entries follow the parameter's actual stored tensor order.

## Validation Rules

- Every group name in `sharded`, `shared`, or `reduce` must be initialized by
  `comm.py`.
- `None` is valid only inside `sharded`.
- Every `reduce` group must span ranks that own the same local parameter shard,
  and the planning wgrad contract must show why their gradients are partial.
- Unknown parameters must not silently fall through. Either annotate them from
  the plan or explicitly mark them as unsharded with a documented default.
