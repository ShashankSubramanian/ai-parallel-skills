# Traceable Collectives

Use this subskill only when full-graph capture is an explicit requirement. It
replaces the public mapping wrappers from
`../parallel-mappings/autograd-collectives.md` with collectives the compiler can
trace, instead of hiding them behind `@torch.compiler.disable`. Under region
compilation, skip to the mutation and synchronization rules at the end of this
file.

## Cost of the Disabled-Wrapper Alternative

Under region compilation, each disabled wrapper is one graph break in the
forward and one in the backward.
A transformer with a tensor-parallel block and a context-parallel attention
typically breaks four or five times per block, multiplied by depth. The compiler
still fuses the compute between breaks, which is most of the available win. Pay
the cost of this subskill only when profiling shows the breaks matter, or when
communication and compute overlap is the goal.

## Support Tiers

Collectives are not equally well supported. Design toward the top of this list,
and treat a hot path that depends on the bottom of it as a reason to stay with
region compilation.

1. All-reduce, all-gather, reduce-scatter. Solid. These are the collectives the
   compiler's overlap and bucketing passes are built around.
2. Broadcast, and gather to a root. Traceable, but less optimized.
3. All-to-all. Workable only in the single-tensor form and only with static
   split sizes. Symbolic split sizes make the output shape data-dependent and
   defeat tracing. The tensor-list form has no functional equivalent. Expect
   little help from the overlap passes.
4. Point-to-point send and receive. Not traceable. See the untraceable
   constructs below.

## Option 1: Functional Collectives

`torch.distributed._functional_collectives` provides out-of-place, traceable
collectives. They return new tensors, which is what functionalization requires,
and the compiler can overlap them with surrounding compute.

Requirements before adopting them:

- Shard sizes must come from configuration, not from run-time discovery. Uneven
  shards are perfectly traceable when their sizes are known at construction:
  padding to a constant maximum, narrowing per rank, and concatenating are all
  static operations. The same code becomes untraceable when the sizes are
  learned from a collective during forward, because every downstream shape is
  then data-dependent. Uneven is not the problem; discovered is.
- Group handles must be resolvable at trace time. Pass a stable group name or
  tag rather than a process-group object threaded through model layers.
- All-to-all must use the single-tensor form with explicit static split sizes,
  not the tensor-list form.

Treat this namespace as private. It carries a leading underscore, it is not
covered by the public deprecation policy, and its implementation has already
been rewritten once. Pin the framework version for any project that depends on
it, and expect to revisit the call sites on upgrade.

## Option 2: Registered Custom Operators

When a collective cannot be expressed functionally, register it as a custom
operator so the compiler treats it as an opaque node with known metadata.

1. Define the operator with `torch.library.custom_op`, declaring mutated
   arguments explicitly. Prefer no mutation at all.
2. Register a fake implementation with `register_fake` that returns correct
   output shapes, dtype, and device without communicating. This is what lets the
   compiler trace through without executing a collective.
3. Register the backward with `torch.library.register_autograd`, mapping each
   forward collective to its dual as documented in
   `../../reference/collectives.md`.
4. Keep the public wrapper signature unchanged so model code and equivalence
   tests need no edits.

A correct fake implementation is mandatory. A wrong output shape there produces
a graph that compiles and then fails or silently corrupts results at run time.

## Mutation Rules

Functionalization cannot see mutation hidden inside a collective. In-place
collectives that write into a caller-supplied tensor are unsafe in traced code
and are a latent aliasing bug even in eager mode when the tensor is a live
activation.

- Clone before an in-place collective, or use the out-of-place form.
- Never write into a tensor that is also an input to the surrounding autograd
  function.
- Declare any remaining mutation in the custom-operator schema.

## Untraceable Constructs

These can never appear in a traced region and must be removed or relocated:

- `dist.all_gather_object`, `dist.broadcast_object_list`, and every other
  object-based collective. They pickle through the host. When shard sizes are
  needed, pass them in from configuration instead of discovering them at run
  time.
- `dist.send`, `dist.recv`, `dist.batch_isend_irecv`, and every other
  point-to-point operation. A traceable collective is a value-producing
  operation with an explicit wait on its result, while a point-to-point
  operation is side-effecting and its matching partner lives in another
  process's program. There is no functional equivalent to reach for. Structure
  the code so point-to-point exchanges sit between compiled regions rather than
  inside one: compile the stage or block, and keep the exchange in eager Python.
- Blocking work queries such as `wait()` on a handle stored in Python state.
- Any collective whose participation depends on a tensor value rather than on
  configuration.
- Host synchronization inside the traced region, including `.item()` on a loss
  or norm.

## Checklist

- The chosen option is recorded with the parallelism plan.
- Every collective on a hot path sits in tier 1 or tier 2, or the choice to stay
  with region compilation is documented.
- Every shard size comes from configuration, not from run-time discovery.
- Every traceable collective is out-of-place or declares its mutation.
- Every custom operator has a fake implementation with correct output metadata.
- Every custom operator registers a backward matching the documented dual.
- No object-based collective remains in a forward path.
- Equivalence tests pass unchanged, because wrapper signatures did not change.
