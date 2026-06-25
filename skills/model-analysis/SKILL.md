---
name: model-analysis
description: Analyzes AI model code before parallelization. Use to identify inputs, outputs, key tensor shapes, weights, saved activations, optimizer memory, and the main bottlenecks.
---

# Model Analysis

Use this skill before choosing parallelism. Keep the output compact enough that
the planning skill can use it directly.

## Trace Shapes

Use `../../reference/model-contract.md`. Find the model entry point, forward
signature, loss, optimizer, precision, checkpointing policy, and representative
input shapes.

Trace major forward ops in order, but group repeated layers or identical op
families. Record one entry per op group unless an individual layer has unusual
shapes or dominates memory. For each group, record the op name, count, input
shapes, output shapes, and logical backward contract.

For example, if the op is a linear layer, the contract is:

```text
forward:
y = x * w + bias
b l e, e f -> b l f
b l f + f -> b l f

backward, where loss is the scalar loss:
dloss/dx = dloss/dy * w^T
b l f, f e -> b l e

dloss/dw = x^T * dloss/dy
e (b*l), (b*l) f -> e f

dloss/dbias = sum_{b,l}(dloss/dy)
b l f -> f
```

dloss/dx (or any gradients w.r.t the activations) are called "dgrads" and dloss/dw are called "wgrads".

Use this contract to estimate (calculate):
- Forward FLOPs : Typically a matmul
- Backward FLOPs : Typically 2x the forward FLOPs
- Weight memory: size(weight) * bytes_per_weight
- Cached activation memory (for backward pass): size(cached activation) * bytes_per_activation

For example, for the linear layer contract above:

- Forward FLOPs: `2 * b * l * e * f`.
- Backward FLOPs: Typically 2x the forward FLOPs, so `4 * b * l * e * f`.
- Weight memory: `e * f * bytes_per_weight`.
- Bias memory: `f * bytes_per_bias`.
- Saved activation memory: x is cached for dloss/dw and hence size(x) * bytes_per_activation

## Group Estimates

Estimate contracts and costs by op layer or group. If layers / groups are
repeated, account for repetition by multiplying the estimates accordingly. For
example:

```text
op group: transformer.mlp.fc1
count: n_layers
logical contract: b l e, e f -> b l f
backward: dloss/dx, dloss/dw, dloss/dbias follow linear contract
per-instance estimates:
  flops (TFLOPs): 2 * b * l * e * f
  weight memory (GB): e * f * bytes_per_weight
  cached activation (GB): b * l * e * bytes_per_activation
total estimates:
  flops (TFLOPs): n_layers * 2 * b * l * e * f
  weight memory (GB): n_layers * e * f * bytes_per_weight
  cached activation (GB): n_layers * b * l * e * bytes_per_activation
```
Note logical contract is a string representation of the contract, but the estimates are calculated.
Expand unusual layers or top memory contributors, if you need to.

## Estimate Memory
Total memory cached is the sum of weights and cached activations.

Account for additional memory:
- Gradients
- Optimizer states

For each trainable weight, assume one gradient tensor with the same shape

Optimizer examples:

- SGD without momentum: `0` extra optimizer tensors.
- SGD with momentum: `1 * parameter_numel * bytes_per_state`.
- Adam or AdamW: `2 * parameter_numel * bytes_per_state`.
- Mixed precision Adam with FP32 master weights:
  `2 * parameter_numel * bytes_per_state + parameter_numel * bytes_per_master_weight`.

## Bottleneck
The bottleneck is the dimension that dominates memory and / or compute. For example, if the memory bottleneck is the cached activation, then the bottleneck dimensions can be any of the cached activation dimensions. Similarly, for FLOPs. These dimensions are the candidate sharding dimensions.


## Final Report

Always print the final report using `output-format.md`. The final report should
contain the model contract table, op group table, memory summary table, and
bottleneck table.

## References

- `output-format.md`
- `../../reference/model-contract.md`
