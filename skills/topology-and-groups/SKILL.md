---
name: topology-and-groups
description: Builds Megatron-style hyperrectangle communication groups and a minimal comm.py API for data, tensor, context, and spatial parallel implementations. Use after the parallelism plan is approved and before inserting collectives.
---

# Topology And Groups

Use this skill to implement communication structure before modifying model
layers. Always create or copy a Megatron-style orthogonal rank generator and use
it from a `comm.py` module.

## Required Inputs

- Approved parallelism plan.
- World size and rank launch method.
- Chosen hyperrectangle dimensions such as `dp`, `tp`, `cp`, `sp1`, and `sp2`.
- Explicit rank order such as `tp-cp-dp` or `sp1-sp2-tp-dp`.
- Existing distributed utilities and initialization path.
- Hardware topology assumptions when available.

## Implementation Steps

1. Implement the rank generator from `rank-generator.md`. This should be copied
   or adapted from Megatron's `generate_masked_orthogonal_rank_groups` and
   `RankGenerator` pattern.
2. Implement a target-project `comm.py` from `comm-module.md`.
3. Put all `torch.distributed.new_group(...)` calls inside `comm.py`; do not
   scatter rank math or group creation through model layers.
4. Build every requested group by token name, for example `dp`, `tp`, `cp`,
   `sp1`, `sp2`, `sp1-sp2`, `tp-cp`, or `dp-sp1-sp2`.
5. Expose only the accessors in `group-accessors.md` to model code.
6. Validate `world_size == product(active_dims)` and every sharded tensor
   dimension is divisible by the matching group size unless padding is planned.
7. Add rank membership tests using `process-groups.md`.

## Required Artifact Shape

The target project should end with this split unless an existing equivalent is
already present:

- `rank_generator.py`: pure rank math, no distributed side effects.
- `comm.py`: distributed initialization, group registry, and accessors.
- Tests: rank generation tests and multi-rank group membership tests.

Keep the API generic by group name. Prefer `get_group("tp")` over one-off
helpers like `get_tensor_parallel_group()` unless the project already uses those
names.

## Hyperrectangle Rule

Treat the process topology as an orthogonal hyperrectangle. A token mask selects
which dimensions vary inside a group; all unmasked dimensions select the group
index. This is the Megatron rule:

- Group token `tp`: ranks vary along tensor parallelism while every other
  coordinate is fixed.
- Group token `dp`: ranks vary along data parallelism while every other
  coordinate is fixed.
- Group token `sp1-sp2`: ranks vary over the 2-D spatial plane while `dp`, `tp`,
  and other coordinates are fixed.
- Group token `tp-cp`: ranks vary over the combined model-parallel plane.

## Output Requirements

- A single source of truth for rank math and process groups.
- A `comm.py` with `get_group`, `get_size`, `get_rank`, `get_root`, and world
  accessors.
- Clear initialization and teardown behavior.
- Shape/world-size assertions close to setup.
- No collectives inserted until group handles are available.
- Single-device behavior preserved.

## References

- `rank-generator.md`
- `comm-module.md`
- `group-accessors.md`
- `process-groups.md`
- `../../reference/collectives.md`
- `../../reference/parallelism-contract.md`
