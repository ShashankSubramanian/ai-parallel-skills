# Parallelism Tradeoffs

Use this file to plan and / or explain the pros and cons of the chosen parallelism strategy.

## Data Parallelism

Pros:

- Simple to reason about.
- Preserves model internals.
- Scales batch throughput when the model fits on one GPU.

Cons:

- Replicates weights, saved activations per microbatch, and often optimizer
  state unless a sharded variant is used.
- Requires gradient synchronization.
- Does not solve layers that individually exceed GPU memory.

## Tensor Parallelism

Pros:

- Reduces per-GPU weight and activation memory for sharded layers.
- Works well for large linear, embedding, vocabulary, convolution, and attention
  head dimensions.

Cons:

- Requires layer rewrites or wrappers.
- Adds collectives on forward or backward paths.
- Shape divisibility and layout assumptions become part of the API.

## Sequence Parallelism

Sequence parallelism is an extension of tensor parallelism. It shards eligible
activations over the sequence dimension using the existing `tp` group; it does
not add another mesh dimension or process group. Keep it disabled unless TP is
already required and the expected activation-memory reduction justifies the
extra layout transitions and bookkeeping.

Pros:

- Reduces activation memory for operations such as normalization, dropout, and
  residual computation.
- Can replace some replicated TP activation regions with sequence shards.

Cons:

- Requires all-gather and reduce-scatter transitions at TP boundaries.
- Applies only where operations preserve independent sequence positions.

## Context Parallelism

Pros:

- Reduces sequence/context activation memory.
- Useful for long-context transformers and other sequence models like ViTs.
- Can compose with tensor or data parallelism.

Cons:

- Attention and recurrent/stateful operations may need specialized
  communication.
- Incorrect key/value or state handling silently changes model semantics.
- Communication can grow with sequence length.

## Spatial Parallelism

Pros:

- Reduces memory for image, volume, weather, fluid, and other grid models.
- Matches operators with local neighborhoods when halo exchange is sufficient.
- Can preserve global batch size while fitting large spatial domains.

Cons:

- Boundary conditions and padding must be explicit.
- Global reductions, normalization, and attention over spatial axes need
  collectives.
- Uneven spatial dimensions can complicate sharding.

## Composed Meshes

Pros:

- Attacks multiple memory categories at once.
- Maps naturally to multi-node hardware.

Cons:

- Increases rank bookkeeping and test surface.
- More collectives can overlap or interfere.
- Debugging wrong-rank or wrong-group bugs becomes harder.
