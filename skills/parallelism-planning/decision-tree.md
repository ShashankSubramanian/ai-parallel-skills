# Parallelism Decision Tree

Pick the simplest plan that fits memory and avoids unnecessary collectives. It does not need to be globally optimal, but it should be reasonable and performant at a high level.

## Workflow

1. Using memory bottlenecks as a guide, see if the model fits per device (this includes all memory usage) with batch size. 
2. If the model fits and the user wants throughput or larger global batch, start with data parallelism. Contract shapes typically look like: `b/dp ...`, where the batch is sharded across `dp` ranks. 
3. If weights, gradients, or optimizer state dominate, and there are no other obvious bottlenecks, try sharding parameters and optimizer state across `dp` ranks (FSDP, ZeRO-style parallelism) before adding tensor parallelism. Contract shapes typically include `e/dp ..` for the weights (and gradients, optimizer states as well)
4. If large linear, attention projection, MLP, embedding, or vocabulary weights dominate, use tensor parallelism over embedding/head/vocabulary dimensions. Contract shapes look like `e f/tp` for individual weight tensors (and gradients, optimizer states as well).
5. If saved activations dominate and scale with sequence/context length, try context or sequence parallelism: `b l/cp e`.
6. If saved activations dominate and scale with spatial grids, try spatial parallelism such as `b h/sp_h w/sp_w c`.
7. If one dimension is insufficient, compose meshes in this order: `dp x tp`, `dp x cp`, `dp x sp`, then `dp x tp x cp` or `dp x tp x sp`. Typically, composition is needed only after data parallelism is not sufficient.

## Good Patterns

- Try to fit patterns in standard data parallel, tensor (row and column parallel), context parallel.
- For spatial parallel, typically halo exchange, or distributed tensor rolls are good patterns. Avoid full spatial all-gathers unless necessary in every block.