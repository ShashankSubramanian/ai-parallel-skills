# Comm Module

Use this subskill to build the target project's `comm.py`. It is the only place
that selects the local device and initializes or destroys `torch.distributed`
process groups for the new distributed implementation. The launcher starts the
processes, but `comm.py` owns distributed initialization inside each process.

## Required State

`comm.py` should keep a small registry:

```python
_COMM_GROUPS: dict[str, dist.ProcessGroup | None] = {}
_COMM_RANKS: dict[str, list[int]] = {}
```

`None` is the group handle for a registered single-device communicator.

If the project needs backend-specific variants, keep them explicit, for example
`_COMM_GROUPS_GLOO`.

## Initialization

Implement an initialization path that:

1. Reads planned dimensions, rank order, and group tokens from config or
   explicit arguments.
2. Validates `world_size == product(active_dims)`.
3. When `world_size == 1` and no distributed launcher is present, skips
   `torch.distributed` initialization, registers every planned token with group
   handle `None` and ranks `[0]`, and returns.
4. Otherwise reads rank and local-rank settings provided by the launcher. For
   CUDA/NCCL, selects the local CUDA device before process-group initialization.
5. Initializes the default process group using the planned backend and launch
   settings.
6. Constructs `RankGenerator(dimensions=..., order=...)`.
7. Iterates over the planned group tokens, calls `rank_gen.get_ranks(token)`,
   creates `dist.new_group(ranks)` for each rank list, and stores only the group
   containing the current rank in `_COMM_GROUPS[token]`.
8. Stores the current rank's global rank list in `_COMM_RANKS[token]`.

Keep the group list data-driven:

```python
groups_to_build = ["dp", "tp", "cp", "sp1", "sp2", "sp1-sp2", "tp-cp"]
```

Choose the actual tokens from the finalized topology handoff in the approved
plan. That handoff must already contain the union of groups required by
operation collectives and planned parameter metadata; live `param.comm_metadata`
does not exist yet because annotation follows topology setup.

## Teardown

Provide `destroy()` or an equivalent cleanup helper that destroys every
subgroup and the default process group created by this module, then clears all
registries. Skip process-group destruction when `torch.distributed` was not
initialized for the single-device path.

## Model Integration

Model code should import accessors from `comm.py`, never instantiate
`RankGenerator` directly. Collectives should pass group handles returned by
`get_group(token)`.
