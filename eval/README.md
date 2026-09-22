# Evaluation Runs

This directory holds immutable baseline cases and self-contained attempts to
parallelize them with the skills in `../skills/`.

The ViT case requires PyTorch and PyYAML to run its baseline trainer.

## Layout

```text
eval/
  assignments.yaml
  cases/
    <case>/
      <model>.py
      train.py
      config.yaml
      prompt.md
  runs/
    <run-id>/
      manifest.yaml
      <model>.py
      train.py
      config.yaml
      prompt.md
      distributed/
      result.yaml
      logs/
```

The files copied from a case are the immutable user inputs. The tested agent
must read the model, configuration, and single-device trainer, leave them
unchanged, and create every output under the run's `distributed/` directory.

## Create a run

1. Choose a unique run ID containing the case, agent, and replicate.
2. Create `eval/runs/<run-id>/`.
3. Copy the selected case's contents directly into that directory.
4. Add `manifest.yaml` with the fields below.
5. Start the agent in the run directory with `prompt.md`.
6. Commit the complete run snapshot, including the generated `distributed/`
   tree, compact logs, and `result.yaml`.

```yaml
run_id: vit-agent-name-r1
case: vit
agent: agent-name
model: model-name-and-version
replicate: 1
base_commit: git-commit-before-the-attempt
status: pending
```

Do not copy another solution into a blind run. A visible implementation of the
same case can leak the answer to the tested agent.

## Required generated interface

Every completed run must contain:

```text
distributed/
  analysis.md
  parallelism-plan.md
  train.py
  launch.sh
  tests/
  implementation-report.md
```

Additional model, communication, optimizer, and mapping files belong in the
same directory. `launch.sh` must provide:

```bash
./distributed/launch.sh --help
./distributed/launch.sh test ...
./distributed/launch.sh train ...
```

`test` runs the agent-generated correctness suite. `train` runs the distributed
training entry point using the copied configuration. The agent creates and
documents this launcher, but the user runs it later on the target hardware.
Scheduler integration and profiling are outside the initial evaluation scope.

## Result record

Store a short `result.yaml` in the run directory:

```yaml
status: passed
baseline_unchanged: true
analysis_and_plan_approved: true
tests:
  world_size_1: passed
  distributed_small: passed
launch:
  generated: true
  test: not_run
  train: not_run
unverified:
  - user-run launch commands
notes: ""
```

Record failures rather than deleting unsuccessful attempts. Avoid committing
checkpoints, datasets, or verbose duplicated logs. After the user runs the
generated launcher, update the same result file with the commands and outcomes.
