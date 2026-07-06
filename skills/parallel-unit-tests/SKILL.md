---
name: parallel-unit-tests
description: Adds correctness tests for AI model parallel implementations. Use to verify that distributed data, tensor, context, or spatial parallel code matches the single-device baseline for forward output, input gradients, and weight gradients.
---

# Parallel Unit Tests

Use this skill after topology, parameter annotations, and mappings are
implemented. Correctness tests must pass before optimization work.

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
2. Initialize the single-device model and distributed model from the same
   logical weights.
3. Shard weights explicitly according to the plan.
4. Validate parameter metadata before shared-weight initialization.
5. Broadcast parameters across their `comm_metadata["shared"]` groups.
6. Feed equivalent logical inputs.
7. Gather or reduce distributed outputs into logical form before comparison.
8. Use the same scalar loss for backward.
9. Gather or reduce gradients into logical form before comparison.

## Assertions

- Output tensors match within dtype-appropriate tolerances.
- Input gradients match within dtype-appropriate tolerances.
- Weight gradients match the corresponding shard or gathered logical tensor.
- Every trainable parameter has valid `comm_metadata`.
- Parameters replicated across `shared` groups match after initialization.
- No rank has missing gradients for trainable local parameters.
- Collectives are called in a consistent order across ranks.

## Tolerances

Use exact or tight tolerances for FP32 deterministic tests. Relax tolerances for
mixed precision, fused kernels, or non-deterministic reductions, but state why.

## Output Requirements

- Add a launchable distributed test.
- Document the command or test marker needed to run it.
- Keep the test small enough for routine development.
- Include at least one failure message that reports rank, shape, and compared
  tensor name.

## References

- `equivalence-tests.md`
- `../../reference/collectives.md`
