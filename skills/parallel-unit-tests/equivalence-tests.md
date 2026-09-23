# Equivalence Tests

Distributed correctness tests should compare logical tensors, not only local
rank shards. The baseline and distributed paths must represent the same
function, full input, full target, scalar loss, and logical parameters.

## Test Lifecycle

Use this order:

1. Initialize the device, default process group when needed, and mesh groups
   through the implementation's `comm.py`. Assert that the mesh product equals
   world size.
2. Seed CPU and accelerator RNGs identically on all ranks.
3. Disable model-parallel group lookup for the baseline path if model
   constructors inspect global communication state.
4. Build the baseline model and full logical metadata, input, and target.
5. Run baseline forward and backward; clone all reference tensors.
6. Restore communication-group lookup.
7. Compute local split shapes before constructing the distributed model. Patch,
   window, padding, and domain rules can make shards uneven.
8. Split global metadata consistently with the input.
9. Build the unwrapped distributed model, validate every referenced group, and
   attach and validate `comm_metadata`.
10. Copy baseline parameters into exact local shards, then initialize and
    validate shared parameters.
11. Install the selected DDP or FSDP wrapper and reduction hooks.
12. Split input and target across each applicable mesh axis.
13. Run distributed forward and backward.
14. Reconstruct and compare output, input gradients, and named weight
   gradients.
15. Call `comm.destroy()` in class/module teardown. Do not directly initialize
    or destroy process groups outside `comm.py`.

Run the baseline on every rank when that keeps the test simple. Because all
ranks use the same seed and model-parallel behavior is disabled, each rank gets
the same oracle without extra broadcasts.

## Identical Logical Weights

Do not rely on matching constructor seeds alone. Explicitly copy each baseline
parameter to the distributed parameter after metadata and wrappers are ready.

For an unsharded parameter, copy the full tensor. For a sharded parameter:

1. Start with the full baseline tensor.
2. For each tensor dimension in `comm_metadata["sharded"]`, get the rank in the
   named group.
3. Slice that dimension using either uniform chunks or the recorded per-rank
   split sizes.
4. Continue slicing the current tensor for additional sharded dimensions.
5. Assert the resulting shape equals the local parameter shape, then copy it
   under `torch.no_grad()`.

Use cumulative offsets for uneven splits:

```python
start = sum(split_sizes[:group_rank])
stop = start + split_sizes[group_rank]
```

Fail on missing or extra parameter names. A warning permits the test to compare
different models accidentally.

After local copying, synchronize parameters marked shared if the implementation
requires it. Verify replicas are equal within each shared group.

## Equivalent Inputs, Targets, and Metadata

Create full logical tensors once. Apply sharding mappings in mesh order:

- Data parallelism commonly splits batch dimension `0`.
- Spatial parallelism commonly splits the last height/width dimensions.
- Context parallelism commonly splits a sequence dimension.
- Tensor parallelism may leave model inputs replicated and shard internal
  activations instead.

Split targets exactly as required by the local output and loss. Split domain
metadata such as area weights, masks, coordinates, or positional tables with
the same boundaries as the corresponding input dimensions.

If split mappings are differentiable custom autograd functions, split the full
input under `torch.no_grad()`, then make the local input a leaf:

```python
with torch.no_grad():
    input_local = split_logical_input(input_full)
input_local = input_local.detach().requires_grad_(True)
```

This avoids comparing the gradient of the global pre-split tensor when the test
intends to inspect the local model input.

## Forward Equivalence

1. Run the baseline model on one device with the full logical input.
2. Run the distributed model on sharded inputs or replicated inputs according to
   the plan.
3. Gather or reduce distributed outputs into the baseline logical shape.
4. Compare values and shapes.

Reconstruct every sharded logical axis, including data-parallel batch shards.
For uneven shards, use split sizes in gather mappings or gather padded tensors
and trim them before concatenation. Gather independent mesh axes in one fixed
order on every rank.

Compare with `torch.testing.assert_close`. Also calculate finite diagnostics:

```python
diff = actual.float() - expected.float()
max_abs = diff.abs().max()
rel_l2 = diff.norm() / expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
```

## Input-Gradient Equivalence

1. Use the same scalar loss for baseline and distributed outputs.
2. Backpropagate once.
3. Gather or reduce input gradients into the baseline logical shape.
4. Compare gradients for every differentiable input.

“Same loss class” is insufficient: the scalar functions must be equivalent.
Check whether a local loss sums or averages over local batch, spatial, token,
channel, or mask elements. Check whether DDP or custom hooks sum or average
gradients. Apply a documented correction factor when reconstructing gradients.
For example, a data-parallel averaged gradient may require division or
multiplication by the data-parallel size depending on the loss reduction and
mapping backward semantics. Derive the factor; do not copy one blindly.

Use the same full target in both paths before splitting. Random targets generated
separately after rank-dependent operations are not equivalent.

## Weight-Gradient Equivalence

1. Start from identical logical weights.
2. After backward, compare each distributed weight gradient to the matching
   baseline slice.
3. If the implementation reduces gradients into replicated form, compare the
   full logical gradient on every relevant rank.
4. Assert that all trainable local parameters have gradients.

Reconstruct a sharded gradient by all-gathering along each tensor dimension
named by `comm_metadata["sharded"]` and concatenating on that same dimension.
Use uneven-size gather logic when shards differ. For replicated parameters,
compare the local gradient directly only after all required
`comm_metadata["reduce"]` reductions have run.

Compare gradients by parameter name first. A flattened all-parameter comparison
is useful as a final aggregate signal, but it can hide wrong name/order
alignment and gives poor failure localization.

For every trainable parameter, assert:

- The baseline and distributed names correspond.
- `comm_metadata` exists and references valid groups.
- A local gradient exists and is finite.
- Its local shape equals the local parameter shape.
- The reconstructed gradient shape equals the baseline parameter shape.
- Values match within the selected tolerance.

## Metadata and Communication Checks

Before numerical comparison:

- Validate all group names in `sharded`, `shared`, and `reduce`.
- Validate that every trainable parameter has metadata.
- Validate that each sharded metadata entry maps to the intended tensor
  dimension.
- Validate uniform divisibility or the exact uneven split table.
- Validate shared parameters after initialization.
- Ensure all ranks take the same branches around collectives. Rank-zero-only
  logging is safe; rank-zero-only collective calls are not.

## Test Data

- Prefer small dimensions that are divisible by the chosen parallel sizes.
- Include at least one non-square or asymmetric shape to catch wrong-axis bugs.
- Use deterministic random inputs unless the model requires structured inputs.
- Include one uneven split when the implementation supports it.
- Toggle optional paths that affect layouts, such as coordinate/position
  embeddings, masks, or alternate heads.
- Keep a realistic-shape case separate or marked slow if it is too expensive
  for routine development.

Parameterize shape, mesh dimensions, feature sizes, optional branches,
tolerance, and expected split behavior. Keep dimensions large enough that no
rank receives an empty shard.

## Launching

The test must be launched with one process per device and environment/config
values that define the same mesh used by the test. Provide a repository-specific
command, for example:

```bash
torchrun --standalone --nproc-per-node=4 -m pytest -s tests/test_parallel_equivalence.py
```

By default, assume SLURM is used with the following launch profile: compute tasks per node from the mesh and
export mesh sizes in the launched process. Suppress duplicate pytest summaries
on nonzero ranks if useful, but never suppress assertion failures.

Example:

```bash
export NVIDIA_TF32_OVERRIDE=0
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  --gpus-per-node="$NPROC_PER_NODE" \
  torchrun --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m pytest -s tests/test_parallel_equivalence.py
```

Include at least:

- A world-size-one launch.
- A launch exercising each sharding axis.
- A combined-mesh launch when the implementation supports multiple axes.

## Failure Diagnostics

Report:

- Rank and group.
- Tensor name.
- Logical shape and local shape.
- Max absolute error and max relative error.
- The collective or mapping boundary most likely involved.
- Mesh sizes, group-local rank, split sizes, dtype, `rtol`, and `atol`.

Use assertion helpers that build these details into the failure message. Assert
shape equality separately before value equality. Assert finiteness before
computing norms so NaNs cannot produce ambiguous failures.

## Common False Positives and False Negatives

- Matching only local outputs can miss a wrong shard order.
- Matching only a flattened weight-gradient vector can pair the wrong
  parameters.
- A relative norm without an absolute tolerance is unstable near zero.
- Independently initialized models can differ despite matching architecture.
- A local mean loss can change gradient scale as mesh size changes.
- Evenly divisible toy shapes do not test uneven patch/domain splitting.
- Leaving model-parallel groups visible while constructing the “baseline” can
  make it distributed too.
- Copying a full parameter directly into a local shard can silently broadcast
  or fail without proving the intended slice.
- Missing gradients skipped by a helper can make an aggregate comparison pass;
  explicitly fail on every missing trainable gradient.

See `test-skeleton.md` for a copy-and-adapt test structure.
