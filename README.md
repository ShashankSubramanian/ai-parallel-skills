# Model Parallel Skills

Agent skills for parallelizing single-device AI model training across GPUs and nodes using **Data (`dp`), Tensor (`tp`), Context (`cp`), and Spatial (`sp`) Parallelism**.

Instead of letting coding agents guess where to place communication collectives, this skill suite enforces a strict, first-principles engineering workflow:
1. **Framework-agnostic analysis & planning:** Mathematical einsum contracts, memory/FLOP bottleneck analysis, device-mesh selection, and forward/backward collective derivation.
2. **PyTorch distributed implementation & verification:** Orthogonal process-group generation (`torch.distributed`), parameter communication metadata, autograd dual mappings, DDP/FSDP reduction hooks, numerical equivalence tests, and optional `torch.compile` readiness.

> **Scope:** Training parallelism (`dp`, `tp`, `cp`, `sp`, plus sequence parallelism over `tp` and DDP/FSDP state sharding). Pipeline parallelism and inference-only parallelism are out of scope.

---

## Key Features

- **Contract-Driven Sharding (Framework-Agnostic):** Forces the agent to write logical and sharded tensor/einsum contracts (`b l e, e f -> b l f`) and mathematically derive forward collectives, activation gradients (`dgrad`), and parameter weight-gradient (`wgrad`) reductions before writing code.
- **Non-Destructive Baseline Contract:** The agent treats your existing model, config, and single-device trainer as immutable inputs and generates the entire parallel implementation inside a standalone `distributed/` directory.
- **Two Human-in-the-Loop Gates:** The agent stops and waits for explicit approval after generating `distributed/parallelism-plan.md` (and again after `distributed/compile-readiness.md` if using `torch.compile`).
- **Rigorous Equivalence Testing:** Includes a mandatory test skeleton that verifies distributed forward outputs, input gradients, and named weight gradients against the single-device baseline using `torch.testing.assert_close`.
- **`torch.compile` Readiness:** Audits distributed PyTorch code for graph breaks, converts runtime shape checks to `torch._check`, hoists communicator queries to `__init__`, and guides choosing between Tier 1 (`@torch.compiler.disable` region compile) and Tier 2 (traceable custom ops / functional collectives).

---

## Quick Start / Installation

### 1. Connect the skills to your coding agent

**Cursor:**
Symlink or copy the `skills/` directory and `AGENTS.md` into your project or user configuration:
```bash
# Inside your target project (or this repo):
mkdir -p .cursor
ln -s /path/to/model-parallel-skills/skills .cursor/skills
cp /path/to/model-parallel-skills/AGENTS.md ./AGENTS.md
```

**Claude Code / Codex / Other Agents:**
Point your agent's instructions (`CLAUDE.md` or `AGENTS.md`) to read `skills/ai-parallelism-master/SKILL.md` whenever asked to parallelize a model.

### 2. Prepare your target workspace

The master workflow requires three baseline files in your target workspace:
1. **Model implementation** (e.g., `vit.py`) containing the `forward` entry point.
2. **Fixed representative config** (e.g., `config.yaml`) with concrete architecture, batch, sequence/spatial, precision, and optimizer settings.
3. **Single-device training script** (e.g., `train.py`) that consumes the config.

### 3. Prompt the agent

Ask your agent to parallelize the model and provide your target hardware constraints:
```text
Parallelize the model in <workspace>/ using `vit.py`, `config.yaml`, and `train.py`.
Target hardware: 8 nodes x 4 A100 GPUs (40 GB/GPU, NVLink intra-node).
Objective: Fit a 1280x1280 input at global batch size 1 and scale to 32–128 GPUs.
```

---

## End-to-End Agent Workflow

```text
[1. Model Analysis]       --> distributed/analysis.md
        |
[2. Parallelism Planning] --> distributed/parallelism-plan.md
        |
   (USER APPROVAL GATE)
        |
[3. Topology & Groups]    --> distributed/rank_generator.py, distributed/comm.py
        |
[4. Annotate Parameters]  --> param.comm_metadata = {sharded, shared, reduce}
        |
[5. Parallel Mappings]    --> distributed/helpers.py, distributed/mappings.py, distributed/model.py
        |
[6. Equivalence Tests]    --> distributed/test_distributed.py, distributed/launch.sh
        |
[7. torch.compile Audit]  --> distributed/compile-readiness.md (Optional, gated)
        |
[8. Final Report]         --> distributed/implementation-report.md
```

---

## Repository Structure

```text
skills/
  ai-parallelism-master/   Authoritative router and phase-gated checklist.
  model-analysis/          Traces tensor contracts, FLOPs, parameter counts, and memory bottlenecks.
  parallelism-planning/    Selects dp/tp/cp/sp mesh, rank order, collectives, and metadata table.
  topology-and-groups/     Megatron-style orthogonal hyperrectangle rank generator and comm.py API.
  annotate-model/          Attaches and validates sharded/shared/reduce metadata on parameters.
  parallel-mappings/       Autograd collective duals, layer sharding patterns, and DDP/FSDP hooks.
  parallel-unit-tests/     Deterministic single-device vs. distributed equivalence test skeleton.
  torch-compile-readiness/ Audits and refactors distributed code for region or full-graph compilation.

reference/
  model-contract.md        Notation for logical tensor shapes and einsum operation contracts.
  parallelism-contract.md  Rules for sharded/local tensors, contracting/free axes, and wgrad duals.
  example-parallelism.md   Reference contracts for MLP, Attention (TP/CP), and Conv2d (TP/SP).
  collectives.md           Collective primitives and forward/backward autograd dual pairs.

eval/
  cases/                   Immutable baseline models, configs, trainers, and prompts (e.g., ViT).
  runs/                    Self-contained agent evaluation snapshots (e.g., vit-reference-run-r1).
  assignments.yaml         Evaluation task tracker.
```

---

## Generated `distributed/` Interface

For every target model, the agent leaves your baseline files untouched and populates `<target-workspace>/distributed/` with a self-contained implementation and Slurm/`torchrun` launcher:

```bash
# View launcher options and supported mesh arguments
./distributed/launch.sh --help

# Run single-rank or multi-rank numerical equivalence tests (forces NVIDIA_TF32_OVERRIDE=0)
./distributed/launch.sh test --nnodes 1 --nproc-per-node 4 --backend nccl --dp 1 --tp 2 --cp 2

# Launch distributed training
./distributed/launch.sh train --nnodes 8 --nproc-per-node 4 --backend nccl --dp 1 --tp 4 --cp 8 --config config.yaml
```

See `eval/runs/vit-reference-run-r1/distributed/` for a complete example of the generated analysis, plan, code, tests, and implementation report for a 32-to-128 GPU Vision Transformer (`tp` + `cp`).
