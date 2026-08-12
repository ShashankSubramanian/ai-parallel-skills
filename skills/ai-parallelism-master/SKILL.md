---
name: ai-parallelism-master-skill
description: Orchestrates AI model parallelism work. Use when user wants to parallelize their AI model. The agent must analyze the target model, plan the parallelism strategy, wait for approval, then implement communication groups, parameter annotations, mappings, and equivalence tests. Pipeline parallelism is out of scope.
---

# AI Parallelism Master

Use this as the entry point for parallelizing an AI model. Pipeline parallelism is out of scope. You MUST follow the workflow checklist below.

## Workflow Checklist

- [ ] Read the target model, training loop, optimizer, distributed launch path (if any), and existing tests (if any).
- [ ] MUST use `../model-analysis/SKILL.md` to produce the model shapes, memory consumed.
- [ ] MUST save the plan: shapes, memory, bottlenecks to an `analysis.md` file
- [ ] MUST use `../parallelism-planning/SKILL.md` to choose the parallelism strategy
- [ ] MUST save the plan: shapes, memory, bottleneck, sharding, collectives,
  parameter metadata handoff, tradeoffs, and tests to an `parallelism-plan.md` file.
- [ ] MUST point user to the above analysis and plan files and stop and wait for explicit user approval.
- [ ] MUST implement communication groups (topology) using `../topology-and-groups/SKILL.md`
- [ ] MUST annotate every trainable parameter with `comm_metadata` using `../annotate-model/SKILL.md`
- [ ] MUST implement helper functions for collectives and mappings using `../parallel-mappings/SKILL.md`
- [ ] MUST add equivalence tests for forward, input gradients, and weight gradients using `../parallel-unit-tests/SKILL.md`.
- [ ] Run or document the correctness tests.

## Skill Routing

- Analysis: `../model-analysis/SKILL.md`
- Planning: `../parallelism-planning/SKILL.md`
- Topology/groups: `../topology-and-groups/SKILL.md`
- Parameter annotations: `../annotate-model/SKILL.md`
- Mappings/collectives: `../parallel-mappings/SKILL.md`
- Tests: `../parallel-unit-tests/SKILL.md`
