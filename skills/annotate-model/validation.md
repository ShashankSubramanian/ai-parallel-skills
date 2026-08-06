# Validation

Use this subskill to verify parameter annotations before training starts.

## Required Checks

Implement a helper such as `validate_param_comm_metadata(model)` that checks:

- Every trainable parameter has `comm_metadata`.
- Metadata has exactly the keys `sharded`, `shared`, and `reduce`.
- `sharded`, `shared`, and `reduce` are lists.
- Every non-`None` group name appears in `comm.get_names()`.
- `None` appears only in `sharded`.
- `len(metadata["sharded"]) == param.ndim`; scalar parameters therefore use
  `[]`, while every unsharded dimension of a non-scalar parameter uses `None`.
- If a parameter is sharded over group `g`, its corresponding tensor dimension
  matches the local shard shape expected by the plan.

## Summary Output

During bring-up, print or log a concise table from rank zero:

```text
parameter name | shape | sharded | shared | reduce
```

Keep the summary optional or debug-gated for production training.

## Test Expectations

Tests should verify:

- No trainable parameters are missing metadata.
- Unknown group names fail validation.
- Shared parameters match across their `shared` groups after initialization.
- Every required extra gradient reduction is named by valid `reduce` metadata.
  The parallel-mappings and parallel-unit-tests stages verify that those
  reductions are actually applied.
