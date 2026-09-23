---
name: ai-parallelism-master-skill
description: Orchestrates AI model training parallelism work. Use when a user wants to parallelize training for an AI model. The agent must analyze the target model, plan the parallelism strategy, wait for approval, then implement communication groups, parameter annotations, mappings, equivalence tests, and a test/train launcher. Inference-only parallelism and pipeline parallelism are out of scope.
---

# AI Parallelism Master

Use this as the entry point for parallelizing AI model training. Inference-only
parallelism and pipeline parallelism are out of scope. You MUST follow the
workflow checklist below.

## Required Model Information Gate

Before beginning analysis, explicitly ask the user to provide or identify all
three input artifacts below and read them in full:

- The model implementation containing the forward entry point.
- One fixed representative configuration file with concrete architecture,
  input, target, precision, optimizer, checkpointing, and dynamic-dimension
  values or ranges.
- The single-device training entry point that consumes that configuration.

Also read any files directly imported by those artifacts that define model or
training semantics. Do not begin analysis while an artifact is missing or while
required values remain unresolved; ask the user instead of inventing defaults.
The analysis must calculate shapes, memory, and FLOPs for the one representative
configuration. Possible future configurations may be considered during
planning, but they do not replace the fixed analysis.

## Required Output Location

Identify the target workspace that contains the supplied model, configuration,
and training entry point. If the artifacts are in different directories or the
workspace is ambiguous, ask the user to choose it. Use
`<target-workspace>/distributed/` for every artifact produced by this workflow.
If `distributed/` already exists, inspect it and ask the user whether to reuse
it or choose a different target workspace. Do not overwrite or delete existing
artifacts without explicit user approval.

Keep all artifacts produced by this workflow in that directory, including:

- `analysis.md`
- `parallelism-plan.md`
- The complete distributed model implementation
- Communication topology, groups, collectives, and mappings
- Distributed training code
- `launch.sh`
- Equivalence and distributed unit tests
- `implementation-report.md`

Do not modify the supplied model, configuration, or single-device training
files. Use them as the immutable baseline and put the complete distributed
implementation and all supporting artifacts under the target workspace's
`distributed/` directory.

## Required Launch Interface

After implementing the distributed training path and correctness tests, create
`<target-workspace>/distributed/launch.sh` as the user-facing entry point. It
MUST:

- support `test` and `train` modes and print usage with `--help`;
- launch the generated distributed correctness tests in `test` mode;
- launch the generated distributed training entry point in `train` mode;
- accept or document the process count, node/rendezvous values, backend, and
  implemented parallel degrees;
- validate that world size and parallel degrees form a supported mesh, with a
  clear error for invalid combinations;

### Launch Profile 

By default, assume SLURM is used with the following launch profile:

- be invoked once inside the allocation and use `srun` to start exactly one
  `torchrun` agent per node (`--ntasks-per-node=1`);
- use `--rdzv-backend=c10d` and the first allocated hostname as the rendezvous
  endpoint. 
- export `NVIDIA_TF32_OVERRIDE=0` before `srun` for equivalence tests and ensure
  Slurm and Shifter propagate it;

The generated structure is:

```bash
export NVIDIA_TF32_OVERRIDE=0 # for equivalence tests
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  --gpus-per-node="$NPROC_PER_NODE" \
  torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    ...
```



## Workflow Checklist

- [ ] Read the supplied model, fixed configuration, single-device training
  entry point, optimizer path, and existing tests (if any).
- [ ] Record the fixed representative model configuration used for all numeric
  analysis.
- [ ] MUST use `../model-analysis/SKILL.md` to produce the model shapes, memory consumed.
- [ ] MUST save the analysis—shapes, memory, and bottlenecks—to
  `distributed/analysis.md` in the target workspace.
- [ ] MUST use `../parallelism-planning/SKILL.md` to choose the parallelism strategy
- [ ] MUST keep data parallelism as an explicit mesh dimension in every plan;
  use `dp=1` when it is a no-op, and scale `dp` by default when additional
  devices and the global batch permit.
- [ ] MUST save the plan: shapes, memory, bottleneck, sharding, collectives,
  parameter metadata handoff, tradeoffs, and tests to
  `distributed/parallelism-plan.md` in the target workspace.
- [ ] MUST point user to the above analysis and plan files and stop and wait for explicit user approval.
- [ ] MUST implement all distributed code under the target workspace's
  `distributed/` directory.
- [ ] MUST implement communication groups (topology) using `../topology-and-groups/SKILL.md`.
- [ ] MUST annotate every trainable parameter with `comm_metadata` using `../annotate-model/SKILL.md`.
- [ ] MUST implement helper functions for collectives and mappings using `../parallel-mappings/SKILL.md`.
- [ ] MUST add equivalence tests for forward, input gradients, and weight gradients using `../parallel-unit-tests/SKILL.md`.
- [ ] Run or document the correctness tests.
- [ ] MUST create `distributed/launch.sh` in the target workspace with the required
  `test`, `train`, and `--help` interface.
- [ ] If the user wants `torch.compile`, MUST use `../torch-compile-readiness/SKILL.md`, and only after the eager correctness tests pass.
  - [ ] MUST save the readiness report: current tier, reachable tier, and
    per-blocker file and line, to `distributed/compile-readiness.md`.
  - [ ] MUST point the user to the readiness report and stop and wait for
    approval of a target tier before editing for compilation.
- [ ] At the end of the workflow, MUST save
  `distributed/implementation-report.md` with the implemented
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
