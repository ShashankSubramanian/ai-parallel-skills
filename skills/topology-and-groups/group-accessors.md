# Group Accessors

Use this subskill to expose the minimal `comm.py` API consumed by model code and
collective helpers.

## Required Accessors

Implement these generic accessors:

```python
def get_names() -> Iterable[str]: ...
def is_initialized(comm_name: str) -> bool: ...
def get_group(comm_name: str): ...
def get_size(comm_name: str) -> int: ...
def get_rank(comm_name: str) -> int: ...
def get_root(comm_name: str) -> int: ...
def get_ranks(comm_name: str) -> list[int]: ...
def get_world_size() -> int: ...
def get_world_rank() -> int: ...
def get_local_rank() -> int: ...
```

## Semantics

- `get_group(comm_name)` raises a clear error if the named group was not built.
- `get_size(comm_name)` returns `1` when distributed is unavailable or the group
  is absent.
- `get_rank(comm_name)` returns `0` when distributed is unavailable or the group
  is absent.
- `get_root(comm_name)` returns the minimum global rank in the group, or `0` for
  the single-rank case.
- `get_ranks(comm_name)` returns the current rank's global rank list for that
  communicator.
- `get_local_rank()` should prefer `LOCAL_RANK` when present, then fall back to
  `world_rank % torch.cuda.device_count()` when CUDA is available.

## Compatibility Shims

Only add named wrappers when existing project code expects them. If added, keep
them thin:

```python
def get_tensor_parallel_group():
    return get_group("tp")
```

Do not duplicate group state for wrapper APIs.
