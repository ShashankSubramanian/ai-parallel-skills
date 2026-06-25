# AI Parallel Skills

Portable skills for AI coding agents that need to add data, tensor, context, or
spatial parallelism to model code.

Start with `skills/ai-parallelism-master/SKILL.md`. The master skill routes the
agent through model analysis, parallelism planning, topology setup, collective
mappings, and equivalence tests.

Pipeline parallelism, profiling, and full example projects are out of scope for
now.

## Layout

```text
skills/
  ai-parallelism-master/   Orchestrates the full workflow.
  model-analysis/          Builds the compact model, shape, and memory plan.
  parallelism-planning/    Selects data, tensor, context, or spatial sharding.
  topology-and-groups/     Defines rank topology and communication groups.
  parallel-mappings/       Inserts collectives and autograd mappings.
  parallel-unit-tests/     Verifies forward, input-gradient, and weight-gradient equivalence.

reference/
  model-contract.md        Logical tensor and operation contract notation.
  parallelism-contract.md  Sharded/local tensor contract notation.
  example-parallelism.md   MLP, attention, and conv sharding examples.
  collectives.md           Collective operations and forward/backward duals.
```

## Workflow

1. Analyze the target model and print a complete model plan.
2. Estimate per-GPU memory for weights, gradients, optimizer state, saved
   activations, temporary buffers, and communication buffers.
3. Choose a parallelization strategy based on the bottleneck.
4. Present the plan, expected memory reduction, pros, cons, and risks.
5. Wait for explicit user approval before editing code.
6. Implement topology/groups, mappings/collectives, and correctness tests.
7. Verify the distributed version matches the single-device version for forward
   outputs, input gradients, and weight gradients.

## Scope

- Data parallelism: shard batches and synchronize or shard training state.
- Tensor parallelism: shard feature, head, channel, projection, or vocabulary
  dimensions.
- Context parallelism: shard sequence/context dimensions while preserving model
  semantics.
- Spatial parallelism: shard image, volume, grid, latitude, or longitude axes.
- Correctness: compare single-device and distributed forward outputs, input
  gradients, and weight gradients.
