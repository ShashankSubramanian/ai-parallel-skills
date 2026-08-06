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
  parameter. These groups must be synchronized after initialization.
- `reduce`: Communication groups whose ranks must all-reduce this parameter's
  gradient in addition to ordinary data-parallel DDP.

## Logical Versus Stored Layout

Operation contracts use mathematical dimension order, while `sharded` always
uses the actual parameter tensor's dimension order. Record the translation in
the planning handoff before annotation. For example, a linear operation may use
logical weight `e f`, but `torch.nn.Linear.weight` is stored as
`(out_features=f, in_features=e)`. Sharding logical `f` therefore produces local
storage `(f/tp, e)` and metadata `["tp", None]`.

## DP Convention

Do not encode ordinary data parallelism in every parameter's metadata. DDP over
`dp` handles replicated model copies and the standard gradient all-reduce. Use
`reduce` only for extra model-parallel reductions such as `sp1-sp2`, `cp`, or
`tp-sp1-sp2`.

## Shape Rules

- `len(sharded)` must equal `param.ndim`, including zero-dimensional scalar
  parameters.
- A parameter of shape `(out_features, in_features)` sharded by output features
  uses `["tp", None]`.
- A parameter of shape `(out_features, in_features)` sharded by input features
  uses `[None, "tp"]`.
- A bias of shape `(features,)` sharded by features uses `["tp"]`.
- A norm weight of shape `(features,)` replicated across model-parallel ranks
  commonly uses `[None]`.

## Validation Rules

- Every group name in `sharded`, `shared`, or `reduce` must be initialized by
  `comm.py`.
- `None` is valid only inside `sharded`.
- Unknown parameters must not silently fall through. Either annotate them from
  the plan or explicitly mark them as unsharded with a documented default.
