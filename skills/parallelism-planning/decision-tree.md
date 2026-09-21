# Parallelism Decision Tree

Pick the simplest plan that fits memory and avoids unnecessary collectives for
the fixed representative model configuration and input. The mesh always has a
`dp` dimension; `dp=1` is its no-op case.

## Workflow

1. Using memory bottlenecks as a guide, see if the representative model and
   input fit per device, including all memory usage.
2. Start with `dp` as the natural scale-out axis. If the model fits, use the
   largest `dp` supported by world size and global batch. Contract shapes
   typically look like `b/dp ...`, where the batch is sharded across `dp`
   ranks.
3. If weights, gradients, or optimizer state dominate, and there are no other
   obvious bottlenecks, consider FSDP before adding tensor parallelism. FSDP
   shards training state across `dp` ranks between computations but leaves the
   logical operation shapes unchanged. Use the first-order FSDP path only when
   parameters need no additional non-DP wgrad reductions; otherwise use DDP
   unless the plan includes a tested target-version-specific FSDP
   communication hook.
4. If large linear, attention projection, MLP, embedding, or vocabulary weights dominate, use tensor parallelism over embedding/head/vocabulary dimensions. Contract shapes look like `e f/tp` for individual weight tensors (and gradients, optimizer states as well).
5. If saved activations dominate and scale with sequence/context length, try
   context parallelism with an independent `cp` mesh dimension. If TP is
   already selected and eligible sequence activations remain a material memory
   bottleneck, consider sequence parallelism. Otherwise keep it disabled to
   avoid unnecessary layout bookkeeping. When enabled, represent it as
   `sequence_parallel=true`, with shapes such as `b l/tp e`, and reuse the
   `tp` group.
6. If saved activations dominate and scale with spatial grids, try spatial parallelism such as `b h/sp_h w/sp_w c`.
7. If model parallelism is needed, choose its minimum dimensions first, then
   assign all remaining devices to `dp`. Compose meshes in this order:
   `dp x tp`, `dp x cp`, `dp x sp`, then `dp x tp x cp` or
   `dp x tp x sp`.
8. At larger device counts, keep the model-parallel dimensions fixed and grow
   `dp` by default for throughput. Increase `tp` only for a larger model-state
   bottleneck, and increase `cp` or `sp` only for a larger input-activation
   bottleneck. Sequence parallelism remains an option on the existing `tp`
   group rather than an independently sized mesh dimension.

## Good Patterns

- Try to fit patterns in standard data parallel, tensor (row and column parallel), context parallel.
- For spatial parallel, typically halo exchange, or distributed tensor rolls are good patterns. Avoid full spatial all-gathers in every block, unless necessary.