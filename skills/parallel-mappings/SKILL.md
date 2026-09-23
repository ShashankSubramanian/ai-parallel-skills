---
name: parallel-mappings
description: Implements required distributed helper primitives and autograd mappings for data, tensor, context, and spatial parallel model code. Use after topology, comm.py process groups, and parameter annotations are defined.
---

# Parallel Mappings

Use this skill to add the distributed building blocks before editing model
layers. Implement the helper primitives and autograd mappings required by the
approved plan.

## Required Inputs

- Approved model and parallelism plan.
- `comm.py` from the topology-and-groups skill.
- Validated `param.comm_metadata` annotations and shared-parameter
  initialization from the annotate-model skill.
- Logical and per-rank tensor shapes.
- Selected data-parallel wrapper (`DDP` or `FSDP`) and its process group.
- Exact forward and backward equivalence requirements.

## Required Artifact Shape

The target project should end with this split unless an existing equivalent is
already present:

- `distributed/helpers.py`: low-level collectives, split-shape utilities, and
  any plan-required spatial helpers.
- `distributed/mappings.py`: `torch.autograd.Function` wrappers and public
  mapping helper functions.
- Optional domain modules such as `distributed/sht.py` only when the approved
  plan needs a domain-specific distributed transform.

## Implementation Steps

1. Implement the helpers required by the approved plan in
   `distributed/helpers.py` using `helpers.md`.
2. Implement the corresponding mappings in `distributed/mappings.py` using
   `autograd-collectives.md`.
3. Use `comm.get_size`, `comm.get_rank`, `comm.get_group`, and `comm.get_root`
   for all communicator access.
4. Keep helper functions no-op for group size one.
5. Implement the selected DDP or FSDP path described in
   `autograd-collectives.md`. This stage owns consuming
   `param.comm_metadata["reduce"]` for non-DP gradient reductions; do not add
   a competing reduction path during parameter annotation.
   For DDP hooks, preserve runtime `dist.GradBucket` and
   `torch.futures.Future[torch.Tensor]` annotations: do not postpone them into
   strings with `from __future__ import annotations`.
6. Insert public mapping helpers at model tensor boundaries identified in the
   approved plan.
7. Add or update correctness tests through the parallel-unit-tests skill.

## Mapping Rules

1. Keep logical tensor shapes documented at module boundaries.
2. Convert to per-rank sharded shapes only where the plan says ownership
   changes.
3. Insert collectives at boundaries, not opportunistically in unrelated code.
4. Use custom autograd mappings when forward and backward collectives differ.
5. Thread `comm_name` through every mapping; do not pass raw process groups into
   model layers.
6. Add shape assertions near every split, gather, reduce, scatter, roll, or
   transpose.

## Required Mappings

Provide the public collective wrappers required by the approved plan.
Identity/all-reduce, all-gather/split, and reduce-scatter/all-gather are the
core mapping patterns. Implement distributed roll and distributed transpose
only when the approved plan uses them.

## Megatron-Style Layer Patterns

- Column-parallel linear: shard output features, optionally gather output.
- Row-parallel linear: shard input features, reduce summed partial outputs.
- Vocab-parallel embedding or output head: shard vocabulary and reduce or gather
  according to the loss path.
- Attention head parallelism: shard attention heads and preserve head-local
  computation when possible.

Use `megatron-patterns.md` for placement details.

## Output Requirements

- Helper primitives exist before layer edits.
- Autograd mapping wrappers exist before layer edits.
- Parameter annotations already exist and are not inferred inside mappings.
- The selected data-parallel path consumes every validated `reduce` entry
  exactly once.
- Each sharded module documents logical shape and per-rank shape.
- Each collective names its process group.
- Backward behavior is explicit for every custom mapping.
- Single-device behavior remains testable.
- Unit tests are added or updated through the parallel-unit-tests skill.

## References

- `helpers.md`
- `autograd-collectives.md`
- `megatron-patterns.md`
- `../torch-compile-readiness/SKILL.md`
- `../../reference/collectives.md`
- `../../reference/parallelism-contract.md`
- `../../reference/example-parallelism.md`
