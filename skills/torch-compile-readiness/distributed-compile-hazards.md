# Distributed Compilation Hazards

Use this subskill to audit the failure modes that appear only when compilation
and collectives are combined. Each one produces a hang, a silent numerical
difference, or a missing gradient rather than a clean error.

## Rank-Divergent Compilation

Compilation is not synchronized across ranks. A rank that recompiles while its
peers are already waiting in a collective stalls the whole group until the
watchdog fires.

- Run a warmup step with production shapes and dtypes before any timed loop, so
  every rank compiles at the same point.
- Keep the recompilation limit identical on every rank.
- Give the process-group timeout enough headroom to cover a cold compile.
- Treat a mid-training recompilation as a bug to be removed, not a cost to be
  absorbed.

## Rank-Dependent Control Flow

Branching on rank inside a forward, such as adding a class token or slicing a
prediction only on the root of a context-parallel group, produces structurally
different graphs on different ranks. This is safe while collectives are eager,
because Python still executes them in a fixed order. It becomes a deadlock the
moment collectives enter the graph, since the scheduler may reorder them
differently per rank.

- Keep every in-graph collective outside rank-dependent branches.
- Prefer masking with a rank-derived buffer over branching, when the cost is
  acceptable.
- Verify the collective order is identical on all ranks after compilation, not
  only before it.

## Eliminated Gradient Edges

Distributed code often keeps a parameter connected to the loss with a
zero-valued term, so that every replica participates in a declared gradient
reduction. A term multiplied by zero is exactly what constant folding is
designed to remove. If the compiler eliminates it, the parameter receives no
gradient on the affected ranks and the reduction silently desynchronizes.

- Assert that every trainable parameter has a non-`None` gradient on every rank
  after a compiled backward step.
- Prefer masking with a buffer over multiplication by a literal zero.
- Move the construct outside the compiled region when it cannot be made robust.
- Add this assertion to the equivalence tests rather than checking it by hand.

## Backward Hooks and Reductions

The gradient reductions installed by `../parallel-mappings/SKILL.md` run in the
autograd engine, outside the compiled forward.

- Keep compiled autograd disabled unless the hooks have been ported. Parameter
  hooks and custom attribute lookups on parameters break the backward graph.
- Read `comm_metadata` during setup and capture the result in the hook closure.
  Do not read custom parameter attributes from traced code.
- Confirm each declared reduction still runs exactly once under compilation.

## Optimizer and Precision

- A hand-written optimizer that is not a `torch.optim.Optimizer` subclass cannot
  participate in a compiled optimizer step or a capturable path. Keep it eager,
  or port it if the step time justifies the work.
- Batch the master-weight and gradient casts of a mixed-precision optimizer with
  the `torch._foreach_*` family before considering compilation.
- Keep loss scaling, gradient clipping, and norm computation outside the traced
  forward, since each introduces host synchronization.

## Dynamic Shapes

- Mark a dimension dynamic explicitly when it will vary, rather than paying for
  a full re-trace on the second distinct value.
- Keep all other dimensions static, which is the normal case for a model whose
  shard shapes come from configuration.
- Remember that uneven shards are still static. Uneven does not mean dynamic.

## Checklist

- A warmup step precedes every timed or profiled loop.
- No in-graph collective sits inside a rank-dependent branch.
- Every trainable parameter has a gradient on every rank after a compiled step.
- Declared gradient reductions run exactly once under compilation.
- Compiled autograd is disabled, or the hooks have been ported to it.
- Dynamic dimensions are marked explicitly, and the rest stay static.
