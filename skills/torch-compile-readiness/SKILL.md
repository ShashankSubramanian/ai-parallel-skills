---
name: torch-compile-readiness
description: Makes distributed model-parallel code compatible with torch.compile. Use after mappings and collectives are implemented, to convert forward-time raises to torch._check, hoist rank and size lookups to construction-time constants, and choose between region compilation and fully traceable collectives.
---

# torch.compile Readiness

Use this skill to audit and fix a distributed model for `torch.compile`. Run it
after `../parallel-mappings/SKILL.md` has produced working collectives and
mappings, and after eager equivalence tests pass. Correctness comes first; never
trade a passing equivalence test for a faster graph.

## Required Inputs

- Implemented `distributed/helpers.py` and `distributed/mappings.py`.
- Model code with collectives already placed at ownership boundaries.
- Passing eager-mode equivalence tests from `../parallel-unit-tests/SKILL.md`.
- The compile target: which module is wrapped, and whether `fullgraph=True` is
  required.
- Which tensor dimensions are static and which vary across steps.

## Choose the Compile Target First

Two strategies exist and they are mutually exclusive. Decide before editing.

1. **Region compilation.** Public mapping wrappers keep the
   `@torch.compiler.disable` decorator required by
   `../parallel-mappings/autograd-collectives.md`. Every collective becomes a
   graph break and compute between collectives is fused. `fullgraph=True` is
   impossible by construction. Compile the repeated block module rather than the
   whole model so one graph is reused across depth. Prefer this strategy unless
   full-graph capture is an explicit requirement.
2. **Fully traceable collectives.** Replace disabled wrappers with registered
   custom operators or functional collectives so the whole step is one graph.
   This is a project, not a cleanup. Follow `traceable-collectives.md`.

Record the chosen strategy alongside the parallelism plan. Do not mix strategies
within one model.

## Implementation Steps

1. Convert forward-time checks to `torch._check` using `graph-safe-guards.md`.
2. Hoist rank, size, and shard-shape lookups into construction-time constants,
   also covered by `graph-safe-guards.md`.
3. Remove tensor mutation and host synchronization from every path that will be
   traced, per `traceable-collectives.md`.
4. Audit rank-dependent control flow, recompilation, and backward hooks using
   `distributed-compile-hazards.md`.
5. Compile the chosen region and inspect the result with
   `torch._dynamo.explain` or `TORCH_LOGS=graph_breaks`.
6. Re-run the equivalence tests from `../parallel-unit-tests/SKILL.md` under
   compilation, not only under eager execution.

## Required Invariants

1. Use `torch._check` for every shape or value check inside a traced forward.
   Do not use Python `assert` or `raise` there.
2. Cache rank, size, and group lookups as attributes during construction. Do not
   call `comm.get_rank`, `comm.get_size`, or `comm.get_group` inside a traced
   forward.
3. Validate configuration in `__init__`, where a normal `raise` is free and
   reports a clear error before any graph is captured.
4. Keep `.item()`, `.tolist()`, `.cpu()`, and Python `int()` or `float()`
   conversions of tensors out of traced forwards.
5. Return new tensors from traced code. Do not mutate a traced function's input
   tensor in place.
6. Keep every `dist.*_object` collective out of the model forward path. These
   pickle through the host and can never be traced.
7. Make every rank reach the same sequence of collectives and recompile at the
   same step. Rank-divergent compilation deadlocks the job.
8. Derive shard shapes from configuration during construction, not from tensor
   shapes observed during forward.

## Verification

- `torch._dynamo.explain(model)(example_input)` reports the expected number of
  graph breaks and no unexpected break reasons.
- Under strategy 2, `torch.compile(model, fullgraph=True)` succeeds.
- Forward output, input gradients, and weight gradients match the eager
  distributed run within dtype-appropriate tolerances.
- Every trainable parameter still receives a gradient on every rank after a
  compiled backward. Deliberate zero-valued gradient edges are the common
  casualty; see `distributed-compile-hazards.md`.
- A recompilation triggered on one rank does not hang the job.
- Step time and peak memory improve, or the compile is not worth its complexity.

## Output Requirements

- The chosen compile strategy is documented with the parallelism plan.
- Traced forwards contain no host synchronization and no data-dependent raise.
- Construction-time constants replace every forward-time topology query.
- Equivalence tests run in both eager and compiled modes.
- Remaining graph breaks are listed with a reason rather than left implicit.

## References

- `graph-safe-guards.md`
- `traceable-collectives.md`
- `distributed-compile-hazards.md`
- `../parallel-mappings/autograd-collectives.md`
- `../parallel-unit-tests/SKILL.md`
