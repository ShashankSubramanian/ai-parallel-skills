# Readiness Report

The torch-compile-readiness skill must begin by producing this report and must
not edit code before the user approves it. The report states the highest
readiness tier the model can reach, the tier it can reach today, and the exact
blockers standing between the two.

Save the report to `compile-readiness.md` next to the parallelism plan.

## Readiness Tiers

| Tier | Name | Meaning |
| ---: | --- | --- |
| 0 | Eager only | No region of the step can be usefully compiled. |
| 1 | Region compile | Blocks or subgraphs compile; collectives stay eager. |
| 2 | Full graph | The whole step captures with `fullgraph=True`. |

A model sits at the highest tier for which every blocker is resolved. Report the
current tier and the reachable tier separately. They differ whenever a blocker
is fixable but not yet fixed.

## Required Sections

### 1. Verdict

| Field | Value |
| --- | --- |
| Current tier | |
| Reachable tier | |
| Recommended target | |
| Compile unit | Whole model, repeated block, or named submodules. |
| Estimated recompiles per run | |

State the recommended target in one sentence, with the reason. Recommending a
lower tier than is reachable is legitimate when the remaining work does not pay
for itself; say so explicitly rather than leaving it implied.

### 2. Blockers

One row per blocker. Order by the tier each one blocks, then by cost.

| Blocker | Location | Blocks tier | Fixable | Fix |
| --- | --- | ---: | --- | --- |
| | `file.py:line` | 1 or 2 | yes, no, or costly | |

Cite a file and line for every blocker. A blocker without a location is a
suspicion, not a finding, and belongs in the open-questions section instead.

### 3. Tier Rationale

Explain, in prose, why the model cannot advance past its current tier. Name the
specific construct, not the category. Common reasons, with the evidence each
one requires:

- **Shape variation.** Report which dimensions vary, how many distinct values
  occur in a representative run, and the resulting recompile count. Genuine tier
  0 evidence is unbounded or high-cardinality shape variation that marking
  dynamic does not resolve. Batch or sequence dimensions varying over a handful
  of known values are not tier 0; they are a recompile budget.
- **Untraceable collectives.** Name the collective and the call site. Note
  whether it is untraceable in principle, such as an object-based or
  point-to-point operation, or only as currently written, such as a tensor-list
  all-to-all that has a single-tensor equivalent.
- **Run-time shape discovery.** Any shard size learned from a collective rather
  than taken from configuration. Distinguish this from uneven shards, which are
  not a blocker when the sizes are known at construction.
- **Host synchronization.** Every `.item()`, `.tolist()`, or Python conversion
  of a tensor inside the intended compile region.
- **Data-dependent control flow.** Branches on tensor values, as distinct from
  branches on rank or configuration, which specialize cleanly.
- **Mutation.** In-place writes to tensors that cross the region boundary.

For every reason given, state what would have to change to remove it. A blocker
with no stated remedy is incomplete.

### 4. Expected Gain

| Tier | Expected benefit | Cost to reach |
| ---: | --- | --- |
| 1 | Fusion between collectives. | |
| 2 | Adds communication and compute overlap. | |

Keep this honest. Region compilation captures most of the available fusion for a
model whose collectives are sparse relative to compute. Full-graph capture pays
off mainly when overlap is the goal. Do not promise a speedup the report cannot
justify.

### 5. Open Questions

List anything that could not be determined from the code: shape distributions
over a real run, whether a collective's shard sizes are truly configuration
derived, or which framework version the project pins. Ask the user rather than
assuming.

## Rules

- Produce the report before editing any file.
- Cite a file and line for every blocker.
- Separate "untraceable in principle" from "untraceable as written". The first
  caps the reachable tier; the second is only a task.
- Do not report a tier without naming what holds the model there.
- Stop after the report and wait for approval of the target tier.
