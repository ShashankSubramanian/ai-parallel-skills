---
name: ai-parallelism-master-skill
description: Orchestrates AI model parallelism work. Use when user wants to parallelize their AI model. The agent must analyze the target model, plan the parallelism strategy, wait for approval, then implement communication groups, parameter annotations, mappings, and equivalence tests. Pipeline parallelism is out of scope.
---

# AI Parallelism Master

Use this as the entry point for parallelizing an AI model. Pipeline parallelism is out of scope. You MUST follow the workflow checklist below.

## Required Model Information Gate

Before beginning analysis, verify that the user has provided or identified some minimal information about the model:

- The model implementation and forward entry point
- The architecture configuration or all fixed model dimensions
- Representative input and target shapes
- Precision, optimizer, and activation-checkpointing settings
- Dynamic dimension ranges and relevant divisibility constraints

Fixed dimensions may be read from model code or configuration files (if the user points to them)

## Required Output Location

Create a new directory at `models/distributed_<model>/`, where `<model>` is the
target model's existing name. Keep all artifacts produced by this workflow in
that directory, including:

- `analysis.md`
- `parallelism-plan.md`
- The complete distributed model implementation
- Communication topology, groups, collectives, and mappings
- Distributed launch or training code
- Equivalence and distributed unit tests

Do not write analysis or planning files at the repository root. Do not place
distributed implementation code into the original model directory or modify the
original single-device implementation. Use the original model as the baseline
and put the new implementation and its supporting code under
`models/distributed_<model>/`.

## Workflow Checklist

- [ ] Read the target model, training loop, optimizer, distributed launch path (if any), and existing tests (if any).
- [ ] MUST use `../model-analysis/SKILL.md` to produce the model shapes, memory consumed.
- [ ] MUST save the analysis—shapes, memory, and bottlenecks—to
  `models/distributed_<model>/analysis.md`.
- [ ] MUST use `../parallelism-planning/SKILL.md` to choose the parallelism strategy
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

## Skill Routing

- Analysis: `../model-analysis/SKILL.md`
- Planning: `../parallelism-planning/SKILL.md`
- Topology/groups: `../topology-and-groups/SKILL.md`
- Parameter annotations: `../annotate-model/SKILL.md`
- Mappings/collectives: `../parallel-mappings/SKILL.md`
- Tests: `../parallel-unit-tests/SKILL.md`
