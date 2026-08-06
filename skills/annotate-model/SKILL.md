---
name: annotate-model
description: Annotates every trainable model parameter with comm_metadata describing sharded, shared, and extra gradient-reduction communication groups. Use after topology/groups are implemented and before inserting activation mappings.
---

# Annotate Model

Use this skill after `topology-and-groups` and before `parallel-mappings`.
The goal is to make parameter ownership explicit before collectives and training
hooks are added.

Every trainable parameter must have:

```python
param.comm_metadata = {
    "sharded": [...],
    "shared": [...],
    "reduce": [...],
}
```

## Required Inputs

- Approved model and parallelism plan.
- Parameter metadata handoff table from the planning skill.
- `comm.py` group names from the topology-and-groups skill.
- Logical operation shapes plus actual global and per-rank parameter tensor
  shapes from the planning handoff.
- Training wrapper strategy, usually DDP over `dp`.

## Implementation Steps

1. Read the model modules and list every trainable parameter by name, shape, and
   owner module.
2. Match each parameter to a row in the planning handoff table.
3. Assign the row's `sharded`, `shared`, and `reduce` metadata exactly, using
   `metadata-contract.md` for validation semantics.
4. Prefer passing `comm_metadata` into local wrapper layers when the project has
   them. Otherwise attach `param.comm_metadata` directly after parameter
   creation.
5. Add a model-level verification helper from `validation.md` that fails if any
   trainable parameter is missing metadata or names an unknown group.
6. Add shared-weight initialization from `shared-init.md`.
7. Verify that every required non-DP weight-gradient reduction is represented
   in `param.comm_metadata["reduce"]`, then hand the validated metadata to the
   parallel-mappings stage.
8. Add or update tests for metadata coverage, shape alignment, group validity,
   and shared initialization. The parallel-mappings stage owns reduction-hook
   implementation and numerical gradient-reduction tests.

## Required Semantics

- `dp` normally owns full replicated model copies. Do not list `dp` in every
  parameter's `shared` or `reduce` metadata just to represent ordinary DDP.
- Put groups in `shared` when each rank in that group stores a full copy of the
  parameter and initialization must be synchronized.
- Put groups in `sharded` when the parameter tensor itself is partitioned along a
  corresponding parameter dimension.
- Put groups in `reduce` when weight gradients must be all-reduced beyond normal
  DDP, for example across `sp1-sp2`, `cp`, or `tp-sp1-sp2`.
- Keep `sharded` aligned with parameter tensor dimensions. Use `None` for a
  tensor dimension that is not sharded.
- Do not copy shard-axis positions from logical einsum notation. Match them to
  the actual parameter layout reported by the target framework; for example,
  `torch.nn.Linear` stores logical `e f` as `(f, e)`.

## Output Requirements

- Every trainable parameter has `comm_metadata`.
- Metadata follows the approved plan and uses only groups built by `comm.py`.
- Every metadata value can be traced back to a planning handoff row.
- Shared parameters are initialized identically across their `shared` groups.
- Every additional non-DP gradient reduction is named in `reduce` metadata,
  ready for the parallel-mappings stage to consume.
- A short annotation summary is added to code comments, tests, or documentation.

## References

- `metadata-contract.md`
- `annotation-patterns.md`
- `shared-init.md`
- `validation.md`
- `../topology-and-groups/SKILL.md`
- `../parallel-mappings/SKILL.md`
- `../parallel-unit-tests/SKILL.md`
