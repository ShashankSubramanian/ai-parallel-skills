---
name: parallelism-planning
description: Chooses a good parallelism strategy from a model analysis. Use to select sharding dimensions, estimate per-GPU memory reduction across GPU counts, explain communication costs, and produce pros and cons before implementation. Pipeline parallelism is excluded.
---

# Parallelism Planning

Use this skill after the model-analysis skill has produced a tensor and memory plan. It helps the user either visualize or choose a parallelism strategy.

## Workflow 

- [ ] Read the model plan and memory summary.
- [ ] Ask the user for preferences for parallelism such as memory per device, global batch size intended for their runs, expected number of devices.
- [ ] If user know what parallelism to use, start from `../../reference/parallelism-contract.md` to write the local tensor shapes, operation contracts, and parameter metadata for the chosen parallelism.
- [ ] If user does not know what parallelism to use, use `decision-tree.md` to suggest a possible plan, write the local tensor shapes, operation contracts, and parameter metadata for the suggested plan.
- [ ] Iterate on plans until user is satisfied, then print the final plan, and ask the user for approval of the plan.

## Output Format for Plan

Use the following format to output the plan:

| Op group | Mesh | Logical | Sharded | Forward Collectives | dgrad Collectives | wgrad Collectives |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.fc1` | `dp x tp` | `b l e, e f -> b l f` | `b/dp l e, e f/tp -> b/dp l f/tp` | none | all-reduce (tp) | all-reduce (dp)|
| `mlp.fc2` | `dp x tp` | `b l f, f e -> b l e` | `b/dp l f/tp, f/tp e -> b/dp l e` | all-reduce (tp)| none | all-reduce (dp)|

Then include the parameter metadata handoff for `../annotate-model/SKILL.md`:

| Parameter pattern | Logical shape | Local shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.fc1.weight` | `e f` | `e f/tp` | `[None, "tp"]` | `[]` | `[]` | row/column choice from the plan |

Rules:

- Any planned parameter sharding must appear in `sharded`, aligned with parameter tensor dimensions.
- Any replicated model-parallel parameter copies that need synchronized initialization must appear in `shared`.
- Ordinary `dp` wgrad all-reduce is handled by DDP and should not appear in `reduce`; non-DP wgrad all-reduces must appear in `reduce`.

## References

- `decision-tree.md`
- `../../reference/collectives.md`
- `../../reference/parallelism-contract.md`
- `../../reference/example-parallelism.md`
