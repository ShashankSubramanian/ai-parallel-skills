# Distributed Helpers

Use this subskill to implement `distributed/helpers.py`. These are the primitive
operations that autograd mappings wrap. Keep them small, synchronous, and
indexed by `comm_name`.

## Required Imports

Helpers should use:

- `torch`
- `torch.distributed as dist`
- The project `comm.py` from topology-and-groups.

## Required Split Utilities

Implement:

```python
def compute_split_shapes(size: int, num_chunks: int) -> list[int]: ...
def compute_split_shapes_for_patching(
    size: int, num_chunks: int, patch_size: int
) -> list[int]: ...
def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    num_chunks: int,
    sections: list[int] | None = None,
) -> list[torch.Tensor]: ...
```

Behavior:

- `compute_split_shapes` should support uneven but balanced splits.
- `compute_split_shapes_for_patching` should split in patch units first, then
  convert back to element sizes and put any remainder on the last shard.
- `split_tensor_along_dim` should validate the dimension and chunk count, use
  explicit `sections` when provided, and return the `torch.split` list.

## Required Collective Helpers

Implement:

```python
def _reduce(input_: torch.Tensor, comm_name: str) -> torch.Tensor: ...
def _split(
    input_: torch.Tensor,
    dim_: int,
    comm_name: str,
    sections: list[int] | None = None,
) -> torch.Tensor: ...
def _gather(
    input_: torch.Tensor,
    dim_: int,
    shapes_: list[int] | None,
    comm_name: str,
) -> torch.Tensor: ...
def _reduce_scatter(
    input_: torch.Tensor,
    dim_: int,
    shapes_: list[int] | None,
    comm_name: str,
) -> torch.Tensor: ...
```

Behavior:

- Return the input unchanged when `comm.get_size(comm_name) == 1`.
- Make tensors contiguous before collectives.
- Use `comm.get_group(comm_name)` for every collective.
- Use `comm.get_rank(comm_name)` to choose local split or output shape.
- `_gather` must support uneven local shapes via `shapes_`.
- `_reduce_scatter` must split along `dim_`, reduce, and keep the caller's local
  shard.

## Required Spatial Helpers

Implement:

```python
def invert_permutation(sigma: list[int]) -> list[int]: ...
def _distributed_roll(
    x: torch.Tensor,
    shift: int,
    shift_dim: int,
    comm_name: str,
) -> torch.Tensor: ...
def _transpose(
    tensor: torch.Tensor,
    dim0: int,
    dim1: int,
    dim0_split_sizes: list[int],
    dim1_split_sizes: list[int],
    comm_name: str,
    async_op: bool = False,
) -> torch.Tensor: ...
```

Behavior:

- `_distributed_roll` should implement a roll over a sharded dimension with
  neighbor exchange and local `torch.roll`.
- `_transpose` should split along `dim0`, use `dist.all_to_all`, and concatenate
  along `dim1`.
- Both helpers must work as no-ops or local operations when group size is one.

## Parameter Metadata Boundary

Do not infer or attach `param.comm_metadata` in mapping helpers. Parameter
annotation and shared-weight initialization belong to
`../annotate-model/SKILL.md`. Helpers may assume metadata validation has already
run when they are called from training hooks.
