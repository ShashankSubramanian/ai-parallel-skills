# Comm Module

Use this subskill to build the target project's `comm.py`. Keep it as the only
place that initializes `torch.distributed` process groups.

## Required State

`comm.py` should keep a small registry:

```python
_COMM_GROUPS: dict[str, dist.ProcessGroup] = {}
_COMM_RANKS: dict[str, list[int]] = {}
```

If the project needs backend-specific variants, keep them explicit, for example
`_COMM_GROUPS_GLOO`.

## Initialization

Implement an initialization path that:

1. Initializes the default process group if the project owns launch setup.
2. Reads planned dimensions and rank order from config or explicit arguments.
3. Validates `world_size == product(active_dims)`.
4. Constructs `RankGenerator(dimensions=..., order=...)`.
5. Iterates over the planned group tokens, calls `rank_gen.get_ranks(token)`,
   creates `dist.new_group(ranks)` for each rank list, and stores only the group
   containing the current rank in `_COMM_GROUPS[token]`.
6. Stores the current rank's global rank list in `_COMM_RANKS[token]`.

Keep the group list data-driven:

```python
groups_to_build = ["dp", "tp", "cp", "sp1", "sp2", "sp1-sp2", "tp-cp"]
```

Choose the actual tokens from the finalized topology handoff in the approved
plan. That handoff must already contain the union of groups required by
operation collectives and planned parameter metadata; live `param.comm_metadata`
does not exist yet because annotation follows topology setup.

## Teardown

Provide `destroy()` or an equivalent cleanup helper that clears registries and
destroys project-owned process groups when appropriate. Do not destroy the
default process group if the training framework owns it.

## Model Integration

Model code should import accessors from `comm.py`, never instantiate
`RankGenerator` directly. Collectives should pass group handles returned by
`get_group(token)`.
