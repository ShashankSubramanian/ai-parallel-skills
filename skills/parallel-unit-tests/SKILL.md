---
name: parallel-unit-tests
description: Adds correctness tests for AI model parallel implementations. Use to verify that distributed data, tensor, context, or spatial parallel code matches the single-device baseline for forward output, input gradients, and weight gradients.
---

# Parallel Unit Tests

Use this skill after topology, parameter annotations, and mappings are
implemented. Correctness tests must pass before optimization work.

## Mandatory Test Skeleton

Before writing or modifying any parallel correctness test, read
`test-skeleton.md` in full and use it as the starting structure for the test.
Do not create the test from scratch or merely consult the skeleton after
implementation. Preserve its setup, baseline/distributed comparison, and
assertion structure unless the target model requires a deviation; document the
reason for every such deviation.

## Required Tests

For a minimal representative model or layer, compare the single-device baseline
with the distributed implementation.

Test:

- Forward output equivalence.
- Input-gradient equivalence.
- Weight-gradient equivalence.
- Shape equivalence after gathering logical outputs.
- Parameter `comm_metadata` coverage and group-name validity.
- Extra gradient reductions named in `comm_metadata["reduce"]`.
- Single-device or world-size-one behavior.

## Test Setup

1. Seed all ranks deterministically.
2. Create one full logical input and target. Every rank running a baseline must
   see identical values; do not add the rank to these seeds.
3. Run the baseline with model-parallel behavior disabled, even if the
   distributed process group is already initialized.
4. Save cloned baseline outputs, differentiable-input gradients, and named
   parameter gradients.
5. Compute local split shapes, including uneven shards caused by patching,
   windows, padding, or sequence boundaries.
6. Build the distributed model with local metadata and validate all required
   communication groups.
7. Attach and validate `comm_metadata`, install distributed wrappers/hooks, and
   copy each full baseline parameter into its exact local shard.
8. Split the full input and target into equivalent local tensors. Detach split
   inputs and re-enable gradients so their `.grad` fields are populated.
9. Run the distributed forward and use a loss mathematically equivalent to the
   baseline scalar loss. Account explicitly for local mean/sum reductions and
   any data-parallel gradient averaging.
10. Reconstruct logical outputs, input gradients, and parameter gradients.
11. Compare named tensors and shapes; use a flattened aggregate gradient check
    only as an additional summary.

Keep the same operation order on every rank. If a baseline model must coexist
with initialized model-parallel groups, temporarily disable only the project
lookup used by baseline constructors and restore it before distributed work.

## Assertions

- Output tensors match within dtype-appropriate tolerances.
- Input gradients match within dtype-appropriate tolerances.
- Weight gradients match the corresponding shard or gathered logical tensor.
- Every trainable parameter has valid `comm_metadata`.
- Parameters replicated across `shared` groups match after initialization.
- No rank has missing gradients for trainable local parameters.
- Collectives are called in a consistent order across ranks.
- The product of mesh dimensions equals world size, and every tested logical
  dimension satisfies the implementation's divisibility or uneven-split rule.
- Baseline and reconstructed tensors are finite before error metrics are used.
- Parameter names and traversal order agree before any aggregate flattening.

## Tolerances

Use exact or tight tolerances for FP32 deterministic tests. Relax tolerances for
mixed precision, fused kernels, or non-deterministic reductions, but state why.
Prefer `torch.testing.assert_close` with both `rtol` and `atol`. A standalone
relative norm can divide by zero and can hide localized errors; report it only
as a diagnostic with a clamped denominator.

## Output Requirements

- Add a launchable distributed test.
- Document the command or test marker needed to run it.
- Keep the test small enough for routine development.
- Include at least one failure message that reports rank, shape, and compared
  tensor name.
- Include a fast, small uneven or asymmetric case and a realistic-shape case
  when routine resources allow it.

## References

- `equivalence-tests.md`
- `test-skeleton.md`
- `../../reference/collectives.md`
