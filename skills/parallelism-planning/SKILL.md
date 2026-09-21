---
name: parallelism-planning
description: Chooses a good parallelism strategy from a model analysis. Use to select sharding dimensions, estimate per-GPU memory reduction across GPU counts, explain communication costs, and produce pros and cons before implementation. Pipeline parallelism is excluded.
---

# Parallelism Planning

Use this skill after the model-analysis skill has produced a tensor and memory plan. It helps the user either visualize or choose a parallelism strategy.

## Workflow 

- [ ] MUST read the fixed representative model configuration, input contract, and
  memory summary from the analysis.
- [ ] MUST ask for the primary device count, devices per node, per-device memory,
  global batch size, and throughput or latency objective. Then ask for the
  additional device counts and input scales the user expects to run. Then ask
  if the user has specific parallelism in mind as well.
- [ ] Keep `dp` as an explicit mesh dimension in every plan, including
  single-device or pure model-parallel cases where `dp=1`. Data parallelism is
  the default scale-out axis.
- [ ] Choose candidate sharding axes and tentative mesh dimensions from the
  bottleneck, device count, and hardware constraints. If the user does not know
  what parallelism to use, use `decision-tree.md` to suggest a candidate.
- [ ] Choose the minimum `tp`, `cp`, or `sp` dimensions needed to fit the
  representative model and input. Keep `sequence_parallel=false` by default.
  Enable it only when TP is already required and sequence-sharded activations
  materially address a remaining memory bottleneck. When enabled, use the `tp`
  group and do not create a separate `seqp` mesh dimension. Assign remaining
  devices to `dp` when the global batch supports that degree.
- [ ] If the user provides future model configurations, evaluate each as a
  separate projection from the fixed analysis. Do not claim a materially
  different model configuration as validated without a separate analysis.
- [ ] Use `../../reference/parallelism-contract.md` to derive local tensor
  shapes, operation contracts, forward/backward collectives, and parameter
  metadata for the candidate mesh.
- [ ] For every trainable parameter, write the wgrad einsum and reapply the
  parallelism-contract sharding rules to derive its `reduce` metadata.
- [ ] Select `DDP` or `FSDP` as the data-parallel wrapper. The first-order FSDP
  path is valid only when every effective non-DP `reduce` list is empty;
  otherwise choose DDP unless the plan explicitly includes a tested
  target-version-specific FSDP communication hook.
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
| Representative model configuration | |
| Memory target per device | |
| Global batch, microbatch, accumulation, and representative inputs | |
| Available devices | `8` |
| Hardware topology assumptions | one 8-GPU node |
| Bottleneck | |
| Candidate mesh | `dp=2, tp=4` |
| Data-parallel wrapper | `DDP` or `FSDP` |
| Sequence parallelism | `false` unless required; reuses `tp` when enabled |

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
- Add a model-parallel group to `reduce` only when its ranks own the same local
  parameter shard and the wgrad einsum shows that they compute partial,
  non-duplicate gradients over a sharded contracting label.
- Ordinary `dp` wgrad all-reduce is handled by DDP and should not appear in `reduce`; non-DP wgrad all-reduces must appear in `reduce`.

## Rank Order Selection

Rank order lists mesh dimensions from the fastest- to slowest-varying
global-rank coordinate. For example, `tp-dp` gives TP groups contiguous global
ranks. Prefer communication-heavy model-parallel dimensions first so their
groups remain within a node when possible. Place `dp` later when data-parallel
groups may span nodes. Explain how the selected rank order maps every group
onto the stated hardware topology.

## Communication Group Token Derivation

Derive the complete group-token list before the topology handoff:

- Include every declared atomic mesh dimension, including `dp`.
- Include every exact group token named by an operation collective or by a
  parameter's `sharded`, `shared`, or `reduce` metadata.
- When `dp > 1`, add a DP-composite candidate for every non-DP token in
  `shared` or `reduce`. Prepend `dp` while preserving the original token, for
  example `cp -> dp-cp` and `tp-cp -> dp-tp-cp`.
- Keep both the original token and its DP-composite candidate. This lets
  implementation choose separate synchronization or one fused synchronization
  without rebuilding topology.
- Do not create a `seqp` token; sequence parallelism reuses `tp`.
- Deduplicate tokens and reject any token containing a dimension absent from
  the planned mesh.

After the operation and parameter decisions are complete, include the finalized
topology handoff for `../topology-and-groups/SKILL.md`:

| Field | Planned value |
| --- | --- |
| World size | `8` |
| Launch method | `torchrun --nproc-per-node=8 ...` |
| Mesh dimensions | `dp=2, tp=4` |
| Sequence parallelism | `false` unless required; reuses `tp` when enabled |
| Rank order | `tp-dp` |
| Group tokens to build | `dp`, `tp` |
| Hardware topology assumptions | one 8-GPU node |
| Divisibility or padding rules | `f % tp == 0`; no padding |

The group-token list must be the complete result of the derivation above, not
an illustrative list. Assert that world size equals the product of active mesh
dimensions. The mesh and group-token list must always include `dp`; use `dp=1`
when no batch replication is possible or needed. Rank order is required even
when two orders would produce the same group sizes. Sequence parallelism must
reuse the `tp` group and must not add a `seqp` mesh dimension or group token.
When enabled, the plan must state the remaining activation bottleneck and the
estimated memory reduction that justifies its additional layout transitions.

Include the expected per-GPU memory at each supported GPU count. Account for
weights, gradients, optimizer state, saved activations, temporary buffers, and
communication buffers; state assumptions when a category is symbolic or
unknown.

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `dp=1` | | | | | | | |
| `8` | `dp=2, tp=4` | | | | | | | |

For each requested scale, state how the mesh changes. Hold the representative
model configuration fixed unless a row is explicitly labeled as a model-size
projection.

| Scale | Model/input configuration | Mesh | Reason for changing model-parallel dimensions |
| --- | --- | --- | --- |
| Primary | representative | `dp=2, tp=4` | minimum TP needed to fit |
| More devices | representative | `dp=4, tp=4` | TP unchanged; scale through DP |
| Larger model projection | explicitly specified larger model | `dp=2, tp=8` | additional TP needed for model state |

Use the following selection rule: choose the minimum model-parallel mesh needed
to fit the model and input. For a throughput objective, assign remaining
devices to `dp` by default. For a latency objective, do not increase `dp` under
the claim that it reduces single-step latency; use only parallel dimensions
that address the latency bottleneck. For every plan, require
`global_batch = microbatch_per_rank * dp * gradient_accumulation_steps`.
Reject a configuration when the mesh dimensions do not multiply to the
selected world size, the batch equation does not hold, or a sharded dimension
is not divisible. Do not silently alter the semantic global batch size.

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
