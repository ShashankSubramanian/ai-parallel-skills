# Collectives

Collectives must be called by every rank in a process group in the same order.

## Common Operations

all-reduce:

- Input: local tensor on every rank.
- Output: reduced tensor on every rank.
- Use for summing partial outputs or synchronizing replicated gradients.

reduce-scatter:

- Input: local tensor or list of chunks on every rank.
- Output: one reduced shard per rank.
- Use when the reduced result should remain sharded.

all-gather:

- Input: local shard on every rank.
- Output: full logical tensor or list of shards on every rank.
- Use when a consumer requires the full logical tensor.

broadcast:

- Input: source rank tensor.
- Output: same tensor on every rank.
- Use for parameter or metadata synchronization.

scatter:

- Input: full logical tensor on source rank.
- Output: shard on each rank.
- Use for initial tensor distribution.

gather:

- Input: local shard on each rank.
- Output: full logical tensor on destination rank.
- Use for diagnostics or tests when all ranks do not need the result.

## Forward and Backward Duals for Differentiable Collectives

- Split forward pairs with all-gather backward.
- All-gather forward pairs with split or reduce-scatter backward.
- All-reduce forward pairs with identity backward.
- Identity forward pairs with all-reduce backward.
- Reduce-scatter forward pairs with all-gather backward.

## Safety Checklist

- Confirm group handle and group size.
- Confirm all ranks use the same tensor shape unless the collective supports
  uneven shapes and the implementation handles them.
- Confirm dtype and device match.
- Confirm collectives are ordered consistently across ranks.
- Make group-size-one behavior a no-op.
