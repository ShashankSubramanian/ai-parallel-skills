# Autograd Collectives

Use this subskill to implement `distributed/mappings.py`. Custom autograd
mappings are required whenever the forward collective and backward collective are
duals.

## Core Autograd Classes

Implement the `torch.autograd.Function` classes required by the approved plan,
backed by `distributed/helpers.py`:

- `_IdentityForwardAllReduceBackward`: forward identity, backward `_reduce`.
- `_AllReduceForwardIdentityBackward`: forward `_reduce`, backward identity.
- `_AllGatherForwardSplitBackward`: forward `_gather`, backward `_split`.
- `_SplitForwardAllGatherBackward`: forward `_split`, backward `_gather`.
- `_ReduceScatterForwardAllGatherBackward`: forward `_reduce_scatter`, backward
  `_gather`.
- `_AllGatherForwardReduceScatterBackward`: forward `_gather`, backward
  `_reduce_scatter`.

When the approved plan requires them, also implement:

- `RollForwardReverseRollBackward`: forward distributed roll, backward reverse
  distributed roll.
- `TransposeForwardTransposeBackward`: forward distributed transpose, backward
  transposes dimensions and split sizes back.

## Public Wrappers

Expose thin functions around `.apply(...)` for every implemented mapping:

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
- Use the PyTorch 2.0 custom-function form with static `forward` and
  `setup_context` methods. Keep `forward` free of the `ctx` argument, and save
  backward state in `setup_context`.
- Do not add `torch.compile` decorators or other compile-specific workarounds
  here. The `../torch-compile-readiness/SKILL.md` stage owns that decision,
  because the correct choice depends on whether the project needs full-graph
  capture. This stage produces mappings that are correct under eager execution.

## DDP Gradient Reduction

The parallel-mappings stage owns implementation and installation of the DDP
communication hook. Registering a hook replaces DDP's default gradient
all-reduce, so the hook must perform both ordinary DP synchronization and every
extra reduction it consumes.

Provide `init_ddp_model_and_reduction_hooks(...)` with this behavior:

1. Inspect only trainable parameters. For each parameter, form its effective
   `reduce` signature by dropping groups whose size is one while preserving the
   deterministic group order from the approved plan.
2. Wrap the model in `DistributedDataParallel` with the `dp` process group.
   For `gloo`, pass `device_ids=None` and `output_device=None`. Use the actual
   DDP argument name `broadcast_buffers`.
3. If every effective signature is empty, install no hook and use DDP's default
   reduction.
4. If every trainable parameter has the same one-token signature `R`, use `R`
   itself when `dp=1`; otherwise require the planned DP-composite group `dp-R`.
   Install a hook that all-reduces the whole bucket once over that group.
   Divide by `dp` size, not by the composite group size. Do not separately
   reduce over `R`.
5. Otherwise install a hook that first all-reduces the whole bucket over `dp`
   and divides by `dp` size. It must then all-reduce the selected bucket
   gradient views once for each exact group named in their parameters'
   effective `reduce` signatures, with no additional DP scaling.

Hook implementation rules:

- Pair `bucket.parameters()` with `bucket.gradients()`; do not branch on
  locally observed `param.grad`.
- Require identical DDP bucket layouts across all ranks participating in an
  extra or fused composite reduction.
- Skip size-one groups before calling `torch.distributed`; a `None` group handle
  denotes a registered singleton, while passing `None` to a collective would
  select the default world group.
- Visit reduction groups in the same planned order on every rank.
- Coalesce selected gradient views with public tensor operations, then copy the
  reduced values back. Do not depend on private flatten/unflatten APIs.
- Chain communication from the DP future and return
  `torch.futures.Future[torch.Tensor]`.
- Ensure parameters with multiple reduction groups are reduced exactly once
  over each group.
- Support unused-parameter handling without changing collective participation
  based on rank-local `None` gradients.
- Do not infer missing metadata. Annotation validation must fail first.

## FSDP Gradient Reduction

When the approved plan selects FSDP:

- Run parameter annotation, validation, and shared initialization before FSDP
  wrapping.
- Wrap with FSDP using the `dp` process group. FSDP owns DP parameter,
  gradient, and optimizer-state sharding; do not encode this transient storage
  sharding in `param.comm_metadata["sharded"]`.
- Do not install the DDP communication hook.
- The first-order supported FSDP path requires every effective non-DP `reduce`
  signature to be empty. If any remains, fail clearly and revise the plan to
  DDP unless a target-version-specific FSDP communication hook is explicitly
  planned, implemented, and equivalence-tested.

## Correctness Expectations

For each mapping, unit tests should compare:

- Forward logical output after any required gather or reduction.
- Gradient with respect to the mapping input.
- Gradient with respect to downstream weights when the mapping wraps a layer.
- Roll and transpose inverses across uneven shard sizes when those mappings are
  implemented.
