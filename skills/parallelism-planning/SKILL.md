---
name: parallelism-planning
description: Chooses a good parallelism strategy from a model analysis. Use to select sharding dimensions, estimate per-GPU memory reduction across GPU counts, explain communication costs, and produce pros and cons before implementation. Pipeline parallelism is excluded.
---

# Parallelism Planning

Use this skill after the model-analysis skill has produced a tensor and memory plan. It helps the user either visualize or choose a parallelism strategy.

## Workflow 

- [ ] Read the model plan and memory summary.
- [ ] Ask the user for preferences for parallelism such as memory per device, global batch size intended for their runs, expected number of devices.
- [ ] Choose candidate sharding axes and tentative mesh dimensions from the
  bottleneck, device count, and hardware constraints. If the user does not know
  what parallelism to use, use `decision-tree.md` to suggest a candidate.
- [ ] Use `../../reference/parallelism-contract.md` to derive local tensor
  shapes, operation contracts, forward/backward collectives, and parameter
  metadata for the candidate mesh.
- [ ] Derive the complete communication-group token set from the collectives
  and parameter metadata, then finalize mesh sizes, rank order, and launch
  method.
- [ ] Validate that the finalized topology supports every planned shard and
  collective and still meets the memory target. Revise the candidate mesh and
  repeat the derivation when it does not.
- [ ] Iterate on plans until user is satisfied, then print the final plan, and ask the user for approval of the plan.

## Output Format for Plan

Use the following format to output the plan:

Start with the constraints, bottleneck, and candidate mesh that drive the
collective reasoning:

| Field | Value |
| --- | --- |
| Memory target per device | |
| Global batch and representative inputs | |
| Available devices | `8` |
| Hardware topology assumptions | one 8-GPU node |
| Bottleneck | |
| Candidate mesh | `dp=2, tp=4` |

Then show how the candidate mesh changes operations and communication:

| Op group | Mesh | Logical | Sharded | Forward Collectives | dgrad Collectives | wgrad Collectives |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp.fc1` | `dp x tp` | `b l e, e f -> b l f` | `b/dp l e, e f/tp -> b/dp l f/tp` | none | all-reduce (tp) | all-reduce (dp)|
| `mlp.fc2` | `dp x tp` | `b l f, f e -> b l e` | `b/dp l f/tp, f/tp e -> b/dp l e` | all-reduce (tp)| none | all-reduce (dp)|

Then include the parameter metadata handoff for `../annotate-model/SKILL.md`:

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mlp.fc1.weight` | `e f` | `e f/tp` | `(f, e)` | `(f/tp, e)` | `["tp", None]` | `[]` | `[]` | logical output-feature shard maps to storage dimension 0 |

Rules:

- Logical operation shapes use the mathematical contract order and may differ
  from the target framework's stored parameter layout. For example,
  `torch.nn.Linear` computes with logical weight `e f` but stores
  `weight.shape == (f, e)`.
- Record both layouts explicitly. `Parameter shape`, `Local parameter shape`,
  and `sharded` must follow the actual tensor returned by
  `named_parameters()`, not the logical einsum notation.
- Any planned parameter sharding must appear in `sharded`, aligned with actual
  parameter tensor dimensions.
- Any replicated model-parallel parameter copies that need synchronized initialization must appear in `shared`.
- Ordinary `dp` wgrad all-reduce is handled by DDP and should not appear in `reduce`; non-DP wgrad all-reduces must appear in `reduce`.

After the operation and parameter decisions are complete, include the finalized
topology handoff for `../topology-and-groups/SKILL.md`:

| Field | Planned value |
| --- | --- |
| World size | `8` |
| Launch method | `torchrun --nproc-per-node=8 ...` |
| Mesh dimensions | `dp=2, tp=4` |
| Rank order | `tp-dp` |
| Group tokens to build | `dp`, `tp` |
| Hardware topology assumptions | one 8-GPU node |
| Divisibility or padding rules | `f % tp == 0`; no padding |

The group-token list must be complete, not illustrative. Include every token
named by an operation collective or by a parameter's `sharded`, `shared`, or
`reduce` metadata. Assert that world size equals the product of active mesh
dimensions. Rank order is required even when two orders would produce the same
group sizes.

Include the expected per-GPU memory at each supported GPU count. Account for
weights, gradients, optimizer state, saved activations, temporary buffers, and
communication buffers; state assumptions when a category is symbolic or
unknown.

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1` | | | | | | | |
| `8` | `dp=2, tp=4` | | | | | | | |

Finish with:

- The bottleneck and why the selected sharding addresses it.
- Pros, cons, and risks, using `tradeoffs.md`.
- Affected files and major model boundaries to change.
- Validation approach, including topology tests and forward, input-gradient,
  and weight-gradient equivalence tests.
- Alternatives considered and why they were rejected.

## References

- `decision-tree.md`
- `tradeoffs.md`
- `../../reference/collectives.md`
- `../../reference/parallelism-contract.md`
- `../../reference/example-parallelism.md`
