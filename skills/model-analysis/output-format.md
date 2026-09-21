# Output Format

The model-analysis skill must end with these tables. Use one row per op group,
not one row per repeated physical layer. Expand only unusual layers or top
memory contributors.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | |
| Entry point | |
| Inputs | |
| Outputs | |
| Parameter dtype | |
| Activation/compute dtype | |
| Optimizer-state dtype | |
| FP32 master weights | |
| Trainable parameters (M) | |
| Optimizer | |
| Checkpointing | |
| Assumptions | |

## Representative Model Configuration

Record the concrete values used for every numeric estimate. Add rows for all
architecture-specific dimensions.

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Layers | | |
| Hidden/channels | | |
| Attention heads | | |
| MLP/intermediate size | | |

## Op Contracts

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| `attn.qkv` | `n_layers` | `b l e`, `e 3e` | `b l 3e` | `b l e, e 3e -> b l 3e` | `dY W^T -> dX`; `X^T dY -> dW` |
| `mlp.fc1` | `n_layers` | `b l e`, `e f` | `b l f` | `b l e, e f -> b l f` | `dY W^T -> dX`; `X^T dY -> dW` |

## Op Costs

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | * | * | * | * | * |
| `mlp.fc1` | `n_layers` | * | * | * | * | * |

Compute each `*` using the fixed representative configuration. A symbolic
formula may accompany the numeric result to show how it scales. For example,
the formula before substituting representative values is:

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | `n_layers * 3 * e * e / 1e6` | `n_layers * 3 * e * e * bytes_per_weight / 1e9` | `n_layers * b * l * e * bytes_per_activation / 1e9` | `n_layers * 2 * b * l * e * 3e / 1e12` | `2 * forward FLOPs` |

For computed shapes (as an example):

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | 50.0 | 0.1 | 0.2 | 0.3 | 0.6 |

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | | | Include bias if present. |
| Gradients | | | Usually one gradient per trainable weight. |
| Optimizer state | | | AdamW usually has two states; FP32 master weights may add more. |
| Saved activations | | | Depends on checkpointing. |
| Total per GPU | | | Sum of the rows above, before sharding. |

This first-order total excludes temporary workspaces, communication buffers,
and framework allocator overhead. It is not a peak runtime-memory estimate.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| | | |

Numeric shape, weight, saved-activation, and FLOP estimates are required for the
representative configuration. If a required model dimension is unknown, report
it as an unmet analysis input instead of completing the analysis with only
symbolic estimates. For implementation-dependent temporary or communication
memory, state that it is excluded from this first-order analysis.
