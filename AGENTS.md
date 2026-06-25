# Agent Guidance

This repository teaches AI coding agents how to add parallelism to their model (includes data and different types of model parallelism). Pipeline parallelism is out of scope.

## Required Workflow

When a user asks for model parallelism, start with the master skill:
`skills/ai-parallelism-master/SKILL.md`.

Do not begin implementation immediately. First produce a plan that includes:

- Model inputs, outputs, major modules, and tensor shapes.
- Which tensors are weights, activations, gradients, optimizer state, activations cached for backward pass, temporary buffers, or communication buffers.
- Per-GPU memory estimates and assumptions.
- The bottleneck driving the parallelization choice.
- The proposed sharding strategy.
- Expected per-GPU memory reduction as the number of GPUs changes.
- Pros, cons, risks, affected files, and validation approach.

Wait for explicit user approval before editing the target model code.

## Skill Routing

- Use `skills/model-analysis/SKILL.md` for model tracing, shape accounting, and
  memory accounting.
- Use `skills/parallelism-planning/SKILL.md` to choose a sharding strategy and
  explain tradeoffs.
- Use `skills/topology-and-groups/SKILL.md` to define ranks, device meshes, and
  communication groups.
- Use `skills/parallel-mappings/SKILL.md` to insert collectives and custom
  autograd mappings.
- Use `skills/parallel-unit-tests/SKILL.md` to prove equivalence with the
  single-device baseline.

Use `reference/` for shared formulas and terminology. Keep skill instructions
portable and avoid assuming a specific IDE / coding workflow.

## Implementation Guardrails

- Preserve the original single-device behavior
- Prefer small, testable changes that expose the parallel topology and sharding
  invariants clearly.
- Treat shape assertions and rank/world-size assertions as required
  documentation, not optional defenses.
- Add correctness tests before optimization work.
- If the model cannot be parallelized safely with the available information,
  stop and ask for the missing model, hardware, optimizer, or training details.
