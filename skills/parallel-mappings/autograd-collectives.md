# Autograd Collectives

Use this subskill to implement `distributed/mappings.py`. Custom autograd
mappings are required whenever the forward collective and backward collective are
duals.

## Required Autograd Classes

Implement these `torch.autograd.Function` classes, backed by
`distributed/helpers.py`:

- `_IdentityForwardAllReduceBackward`: forward identity, backward `_reduce`.
- `_AllReduceForwardIdentityBackward`: forward `_reduce`, backward identity.
- `_AllGatherForwardSplitBackward`: forward `_gather`, backward `_split`.
- `_SplitForwardAllGatherBackward`: forward `_split`, backward `_gather`.
- `_ReduceScatterForwardAllGatherBackward`: forward `_reduce_scatter`, backward
  `_gather`.
- `_AllGatherForwardReduceScatterBackward`: forward `_gather`, backward
  `_reduce_scatter`.
- `RollForwardReverseRollBackward`: forward distributed roll, backward reverse
  distributed roll.
- `TransposeForwardTransposeBackward`: forward distributed transpose, backward
  transposes dimensions and split sizes back.

## Required Public Wrappers

Expose thin functions around `.apply(...)`:

```python
def identity_forward_allreduce_backward(input_, comm_name): ...
def allreduce_forward_identity_backward(input_, comm_name): ...
def allgather_forward_split_backward(input_, dim, shapes, comm_name): ...
def split_forward_allgather_backward(input_, dim, shapes, comm_name): ...
def reducescatter_forward_allgather_backward(input_, dim, shapes, comm_name): ...
def allgather_forward_reducescatter_backward(input_, dim, shapes, comm_name): ...
def roll_forward_reverseroll_backward(x, shifts, dims, comm_names): ...
def transpose_forward_transpose_backward(
    x, dims, dim0_split_sizes, dim1_split_sizes, comm_name
): ...
```

The wrappers are the only functions model code should call directly.

## Implementation Checklist

- Save only what backward needs: `comm_name`, split dimension, split shapes,
  shifts, or transpose dimensions.
- Assert tensor dimension divisibility before split or reduce-scatter unless
  uneven `shapes` are explicitly passed.
- Preserve dtype and device.
- Make operations no-ops when group size is one.
- Ensure every rank calls collectives in the same order.
- Keep asynchronous collectives out of the first implementation unless the
  project already has a safe pattern.
- Return `None` for all non-tensor backward arguments.

## DDP Reduction Hooks

When parameters carry `comm_metadata`, provide a helper equivalent to
`init_ddp_model_and_reduction_hooks(...)`:

- Wrap the model in `DistributedDataParallel` using the `dp` group.
- If model-parallel groups exist, register a comm hook that performs the normal
  data-parallel all-reduce and any extra all-reduces named by each parameter's
  metadata `reduce` list.
- Scale gradients by data-parallel size, not by combined model-parallel group
  size.
- Keep backend-specific DDP arguments explicit, especially for `gloo`.
- Do not infer missing metadata in the hook. Missing metadata should fail during
  the annotate-model validation phase.

## Correctness Expectations

For each mapping, unit tests should compare:

- Forward logical output after any required gather or reduction.
- Gradient with respect to the mapping input.
- Gradient with respect to downstream weights when the mapping wraps a layer.
- Roll and transpose inverses across uneven shard sizes.
