---
name: ai-parallelism-master-skill
description: Orchestrates AI model training parallelism work. Use when a user wants to parallelize training for an AI model. The agent must analyze the target model, plan the parallelism strategy, wait for approval, then implement communication groups, parameter annotations, mappings, and equivalence tests. Inference-only parallelism and pipeline parallelism are out of scope.
---

# AI Parallelism Master

Use this as the entry point for parallelizing AI model training. Inference-only
parallelism and pipeline parallelism are out of scope. You MUST follow the
workflow checklist below.

## Required Model Information Gate

Before beginning analysis, explicitly ask the user to provide or identify the
following minimal information about the model. Do not begin analysis until each
item has been supplied by the user or located in user-identified model code or
configuration files:

- The model implementation and forward entry point
- One fixed representative architecture configuration, with concrete values
  for every model dimension used in the analysis
- Representative input and target shapes
- Precision, optimizer, and activation-checkpointing settings
- Dynamic dimension ranges and relevant divisibility constraints

Fixed dimensions may be read from model code or configuration files if the user
points to them. The analysis must calculate shapes, memory, and FLOPs for this
one representative configuration. Possible future model configurations may be
considered during planning, but they do not replace the fixed analysis.

## Required Output Location

Use `models/distributed_<model>/`, where `<model>` is the target model's existing
name, for the distributed implementation. If the directory does not exist,
create it. If it already exists, inspect it and ask the user whether to reuse it
or choose a different directory. Do not overwrite or delete existing artifacts
without explicit user approval.

Keep all artifacts produced by this workflow in that directory, including:

- `analysis.md`
- `parallelism-plan.md`
- The complete distributed model implementation
- Communication topology, groups, collectives, and mappings
- Distributed launch and training code
- Equivalence and distributed unit tests
- `implementation-report.md`

Do not write analysis or planning files at the repository root. Do not place
distributed implementation code into the original model directory or modify the
original single-device implementation. Use the original model as the baseline
and put the new implementation and its supporting code under
`models/distributed_<model>/`.

## Workflow Checklist

- [ ] Read the target model, training loop, optimizer, distributed launch path (if any), and existing tests (if any).
- [ ] Record the fixed representative model configuration used for all numeric
  analysis.
- [ ] MUST use `../model-analysis/SKILL.md` to produce the model shapes, memory consumed.
- [ ] MUST save the analysis—shapes, memory, and bottlenecks—to
  `models/distributed_<model>/analysis.md`.
- [ ] MUST use `../parallelism-planning/SKILL.md` to choose the parallelism strategy
- [ ] MUST keep data parallelism as an explicit mesh dimension in every plan;
  use `dp=1` when it is a no-op, and scale `dp` by default when additional
  devices and the global batch permit.
- [ ] MUST save the plan: shapes, memory, bottleneck, sharding, collectives,
  parameter metadata handoff, tradeoffs, and tests to
  `models/distributed_<model>/parallelism-plan.md`.
- [ ] MUST point user to the above analysis and plan files and stop and wait for explicit user approval.
- [ ] MUST implement all distributed code under
  `models/distributed_<model>/`.
- [ ] MUST implement communication groups (topology) using `../topology-and-groups/SKILL.md`.
- [ ] MUST annotate every trainable parameter with `comm_metadata` using `../annotate-model/SKILL.md`.
- [ ] MUST implement helper functions for collectives and mappings using `../parallel-mappings/SKILL.md`.
- [ ] MUST add equivalence tests for forward, input gradients, and weight gradients using `../parallel-unit-tests/SKILL.md`.
- [ ] Run or document the correctness tests.
- [ ] If the user wants `torch.compile`, MUST use `../torch-compile-readiness/SKILL.md`, and only after the eager correctness tests pass.
  - [ ] MUST save the readiness report: current tier, reachable tier, and
    per-blocker file and line, to `models/distributed_<model>/compile-readiness.md`.
  - [ ] MUST point the user to the readiness report and stop and wait for
    approval of a target tier before editing for compilation.
- [ ] At the end of the workflow, MUST save
  `models/distributed_<model>/implementation-report.md` with the implemented
  strategy and parallel degrees, files produced, tests run and their results,
  supported and unsupported configurations, remaining assumptions, and every
  unverified claim.
- [ ] MUST point the user to the implementation report and summarize its
  completion status.

## Skill Routing

Before beginning each phase, MUST read the corresponding `SKILL.md` below in
full and follow it. This routing list and the master checklist are not
substitutes for reading the phase skill.

- Analysis: `../model-analysis/SKILL.md`
- Planning: `../parallelism-planning/SKILL.md`
- Topology/groups: `../topology-and-groups/SKILL.md`
- Parameter annotations: `../annotate-model/SKILL.md`
- Mappings/collectives: `../parallel-mappings/SKILL.md`
- Tests: `../parallel-unit-tests/SKILL.md`
- `torch.compile` readiness: `../torch-compile-readiness/SKILL.md`
