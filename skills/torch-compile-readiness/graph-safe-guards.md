# Graph-Safe Guards and Constant Hygiene

Use this subskill to remove the two most common sources of graph breaks and
unwanted specialization: forward-time raises and forward-time topology queries.

## Why Forward-Time Raises Break Compilation

A Python `assert` or `raise` on a tensor shape forces Dynamo to decide the
comparison at trace time. With static shapes it silently bakes in a guard that
recompiles when the shape changes. With dynamic shapes it either specializes the
dimension, discarding the dynamism, or fails outright on a data-dependent value.
`torch._check` states the same constraint to the shape solver, keeps the check
in the graph as a runtime assertion, and never forces specialization.

## Converting Checks

Replace tuple or element shape comparisons with one `torch._check` per
constrained dimension. Keep the message in a lambda so it costs nothing unless
the check fails.

```python
# Before: specializes every dimension and raises during tracing.
expected = (x.shape[0], self.in_channels, self.height, self.width)
if tuple(x.shape) != expected:
    raise ValueError(f"expected input shape {expected}, got {tuple(x.shape)}")

# After: constrains only the dimensions that are actually fixed.
torch._check(
    x.shape[1] == self.in_channels,
    lambda: f"expected {self.in_channels} channels, got {x.shape[1]}",
)
torch._check(x.shape[2] == self.image_height, lambda: "unexpected input height")
torch._check(x.shape[3] == self.image_width, lambda: "unexpected input width")
```

Rules for the conversion:

- Constrain only dimensions the model genuinely requires. A comparison of a
  dimension against itself, such as a batch dimension copied from the same
  tensor, is dead code; delete it rather than translating it.
- Use `torch._check_is_size` for a value used as a size or index, so the solver
  knows it is non-negative.
- Keep checks that compare two symbolic sizes. `torch._check(a.shape[1] ==
  b.shape[1], ...)` teaches the solver a genuine relationship.
- Leave checks inside `@torch.compiler.disable` helpers as ordinary raises while
  region compilation is the chosen strategy. Convert them when, and only when,
  those helpers become traceable.

## Where Validation Belongs

Move every check that depends only on configuration into `__init__`.
Divisibility of features by tensor-parallel size, of a patch grid by
context-parallel size, or of heads by ranks is known before the first tensor
exists. Validation there
raises a clear error at construction, costs nothing per step, and never appears
in a graph.

Keep in the forward only those checks that constrain incoming tensors, and
express those with `torch._check`.

## Constant Hygiene

Every topology query inside a forward is a global lookup that Dynamo must guard.
Resolve them once during construction.

```python
# In __init__, after comm groups exist.
self.tp_size = comm.get_size("tp")
self.cp_size = comm.get_size("cp")
self.cp_rank = comm.get_rank("cp")
self.local_dim = dim // self.tp_size
```

Apply the same treatment to derived quantities:

- Shard shape lists such as per-rank sequence lengths or split heights. Compute
  them from configuration in `__init__` and store them; do not rebuild them in
  forward from observed tensor shapes.
- Lists built by multiplying an observed shape by a group size. A construct such
  as `[x.shape[1]] * comm.get_size("tp")` reads a symbolic size for a value that
  configuration already determines. Store the constant instead.
- Flattened parameter views recomputed on every call. Reshaping a weight for a
  fused projection is cheap but adds graph nodes; hoist it when the shape is
  fixed.

## Checklist

- No `assert` or `raise` on a tensor shape or value inside a traced forward.
- One `torch._check` per genuinely constrained dimension, with a lambda message.
- Configuration validation raised in `__init__`.
- Rank, size, group, and shard-shape lookups cached as attributes.
- No shard shape derived from a tensor shape observed during forward.
- No `.item()`, `.tolist()`, `.cpu()`, `int()`, or `float()` on a traced tensor.
