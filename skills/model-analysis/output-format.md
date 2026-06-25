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
| Dtype | |
| Optimizer | |
| Checkpointing | |
| Assumptions | |

## Ops

| Op | Count | Contract | Weights (GB) | Cached activations (GB) | FLOPs (TFLOPs) |
| --- | ---: | --- | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | `b l e, e 3e -> b l 3e` | * | * | * |
| `mlp.fc1` | `n_layers` | `b l e, e f -> b l f` | * | * | * |

with * computed if shapes are known, else symbolic.
Example, for symbolic shapes:

| Op | Count | Contract | Weights | Cached activations | FLOPs |
| --- | ---: | --- | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | `b l e, e 3e -> b l 3e` | `n_layers * 3 * e * e * bytes_per_weight` | `n_layers * b * l * e * bytes_per_activation` | `n_layers * 2 * b * l * e * 3e` |

For computed shapes (as an example):

| Op | Count | Contract | Weights (GB) | Cached activations (GB) | FLOPs (TFLOPs) |
| --- | ---: | --- | ---: | ---: | ---: |
| `attn.qkv` | `n_layers` | `b l e, e 3e -> b l 3e` | 0.1 | 0.2 | 0.3 |

## Memory Summary

| Category | Estimate | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | | | Include bias if present. |
| Gradients | | | Usually one gradient per trainable weight. |
| Optimizer state | | | AdamW usually has two states; FP32 master weights may add more. |
| Saved activations | | | Depends on checkpointing. |
| Total per GPU | | | Before sharding. |

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| | | |

Keep formulas symbolic when exact sizes are unknown. When sizes are known, add
numeric estimates in GB and TFLOPs.
