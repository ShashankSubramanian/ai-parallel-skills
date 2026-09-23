# ViT Training Analysis

All numeric estimates use the unchanged `config.yaml` in this target workspace.
The supplied `vit.py`, `config.yaml`, and `train.py` are the immutable baseline.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Entry point | `vit.VisionTransformer.forward(x)`; training entry point `train.main()` |
| Inputs | FP32 synthetic images, `(b, 71, 720, 1440)`, fixed `b=1` |
| Targets and loss | Integer class target `(b,)`; FP32 `cross_entropy(logits.float(), target)` |
| Outputs | Logits `(b, 10)` |
| Parameter dtype | FP32 (the model is created in default dtype and moved to CUDA before autocast) |
| Activation/compute dtype | BF16 for eligible CUDA autocast operations; input generation and loss are FP32 |
| Optimizer-state dtype | FP32 AdamW first and second moments |
| FP32 master weights | No separate master copy; the model weights themselves are FP32 |
| Trainable parameters (M) | 568.036362 |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.01` |
| Checkpointing | Disabled |
| Dynamic dimensions | None in the representative configuration; image size and batch are fixed |
| Assumptions | CUDA A100 execution; fused SDPA is available for the stated BF16/head shape. Temporary workspaces, allocator overhead, and communication buffers are excluded from the baseline total. |

## Representative Model Configuration

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Batch / samples | `1` / `1` | `config.yaml:data` |
| Input channels and image | `71`, `720 x 1440` | `config.yaml:model` |
| Patch size and patch grid | `2 x 2`, `360 x 720` | `config.yaml:model` and calculation |
| Patch tokens / sequence length | `259,200` / `259,201` including CLS | Calculation |
| Layers | `24` | `config.yaml:model.depth` |
| Hidden size | `1,024` | `config.yaml:model.dim` |
| Attention heads / head size | `16` / `64` | Configuration and calculation |
| MLP intermediate size | `4,096` | `dim * mlp_ratio` |
| Classes | `10` | `config.yaml:model.num_classes` |
| Dropout | `0.0` | `config.yaml:model.dropout` |
| Epochs | `1` | `config.yaml:training.epochs` |
| Global batch / microbatch / accumulation | `1 / 1 / 1` | Baseline training path |
| Precision | BF16 autocast with FP32 parameters, gradients, optimizer, and loss | `train.py` |

## Shape Trace and Operation Contracts

Notation: `b=1`, `l=259201`, `p=259200`, `e=1024`, `a=16`,
`d=64`, and `f=4096`.

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| `patch_embed` | 1 | `b 71 720 1440`; weight `1024 71 2 2` | `b 1024 360 720`, then `b p e` | stride-2 `conv2d` then flatten/transpose | Conv dgrad; input patches contracted with output gradient for wgrad |
| CLS/position/dropout | 1 | `b p e`, `1 1 e`, `1 l e` | `b l e` | concatenate CLS; add position; dropout is identity at `p=0` | Split/add reductions; position and CLS gradients follow their broadcast dimensions |
| `blocks.*.norm1` | 24 | `b l e` | `b l e` | LayerNorm over `e` | LayerNorm dgrad and affine wgrads reduced over `b,l` |
| `blocks.*.attn.qkv` | 24 | `b l e`; logical weight `e 3e` | `b l 3e`, then Q/K/V `b a l d` | `b l e, e 3e -> b l 3e` | `dY W^T -> dX`; `X^T dY -> dW`; bias sum over `b,l` |
| attention scores/value | 24 | Q/K/V `b a l d` | context `b a l d` | `Q K^T -> b a l l`; softmax over keys; `P V -> b a l d` | SDPA backward for Q, K, and V |
| `blocks.*.attn.proj` | 24 | `b l e`; weight `e e` | `b l e` | `b l e, e e -> b l e` | Linear dgrad, wgrad, and bias reduction |
| `blocks.*.norm2` | 24 | `b l e` | `b l e` | LayerNorm over `e` | LayerNorm dgrad and affine wgrads reduced over `b,l` |
| `blocks.*.mlp.fc1` | 24 | `b l e`; logical weight `e f` | `b l f` | `b l e, e f -> b l f` | Linear dgrad, wgrad, and bias reduction |
| GELU/dropout | 24 | `b l f` | `b l f` | GELU; dropout is identity at `p=0` | GELU derivative |
| `blocks.*.mlp.fc2` | 24 | `b l f`; logical weight `f e` | `b l e` | `b l f, f e -> b l e` | Linear dgrad, wgrad, and bias reduction |
| residual additions | 48 | two `b l e` tensors | `b l e` | elementwise add | Gradient copied to both inputs |
| final norm / head | 1 | `b l e`, then CLS `b e`; logical head `e 10` | `b 10` | LayerNorm; select token 0; `b e, e 10 -> b 10` | Head dgrad/wgrad; scatter CLS dgrad; LayerNorm backward |

## Parameter Count

| Parameter group | Parameters (M) |
| --- | ---: |
| Patch embedding | 0.291840 |
| CLS token | 0.001024 |
| Position embedding | 265.421824 |
| 24 attention QKV projections | 75.571200 |
| 24 attention output projections | 25.190400 |
| 48 block LayerNorms | 0.098304 |
| 24 MLP first projections | 100.761600 |
| 24 MLP second projections | 100.687872 |
| Final norm and classifier | 0.012298 |
| **Total** | **568.036362** |

## Operation Costs

FLOPs count multiply-add as two operations. Backward is estimated as twice
forward, as required by the analysis method. LayerNorm, GELU, softmax,
residual, and data-movement FLOPs are omitted, so the totals are first-order
matmul/convolution estimates. GB is decimal.

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch convolution | 1 | 0.291840 | 0.001167 | 0.294451 FP32 input | 0.150759 | 0.301518 |
| CLS + position embedding | 1 | 265.422848 | 1.061691 | 0.530844 token stream | negligible | negligible |
| Attention QKV linear | 24 | 75.571200 | 0.302285 | 12.740248 | 39.138040 | 78.276081 |
| Attention QK and PV matmuls | 24 | 0 | 0 | 51.359123 fused Q/K/V/output/LSE estimate | 6,604.569811 | 13,209.139623 |
| Attention output linear | 24 | 25.190400 | 0.100762 | 12.740248 | 13.046013 | 26.092027 |
| MLP first linear | 24 | 100.761600 | 0.403046 | 12.740248 linear input | 52.184054 | 104.368108 |
| GELU | 24 | 0 | 0 | 50.960990 pre-activation | omitted | omitted |
| MLP second linear | 24 | 100.687872 | 0.402751 | 50.960990 GELU output | 52.184054 | 104.368108 |
| Block norms | 48 | 0.098304 | 0.000393 | 25.580028 inputs/statistics | omitted | omitted |
| Final norm and head | 1 | 0.012298 | 0.000049 | 0.530846 | 0.00000002 | 0.00000004 |
| **Matmul/convolution total** |  |  |  |  | **6,761.272732** | **13,522.545465** |

The estimated forward-plus-backward work is **20,283.818197 TFLOPs
(20.284 PFLOPs) per sample/optimizer step**. Attention QK/PV matmuls account
for about 97.68% of the counted forward FLOPs.

## Saved-Activation Estimate

This estimate assumes a fused/flash-style SDPA backward that saves Q, K, V,
output, and log-sum-exp rather than materializing the full attention matrix.

| Saved category | Estimate (GB) |
| --- | ---: |
| FP32 image input for patch-convolution wgrad | 0.294451 |
| Stem token stream | 0.530844 |
| Block norm inputs/statistics | 25.580028 |
| QKV linear inputs | 12.740248 |
| Fused SDPA Q/K/V/output/LSE | 51.359123 |
| Attention output-projection inputs | 12.740248 |
| MLP first-linear inputs | 12.740248 |
| GELU inputs | 50.960990 |
| MLP second-linear inputs | 50.960990 |
| Final norm/head | 0.530846 |
| **Total** | **218.438015** |

A dense BF16 attention probability tensor for just one layer would be
`1 x 16 x 259201 x 259201`, or **2,149.925 GB**. If PyTorch selects a math
SDPA backend that materializes this tensor, the representative case is not
feasible on the target hardware. Backend/kernel availability must therefore be
verified on the A100 environment.

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 2.272145 | Position length; transformer hidden/MLP sizes | FP32, including bias |
| Gradients | 2.272145 | Same as parameters | One FP32 gradient per parameter |
| Optimizer state | 4.544291 | Parameter count | Two FP32 AdamW moments; no separate master weights |
| Saved activations | 218.438015 | Sequence length, layers, MLP width | No checkpointing; fused-SDPA assumption |
| **Total per GPU before sharding** | **227.526597** | Sequence length | Not a peak-runtime estimate |

The total excludes temporary attention/matmul workspaces, communication
buffers, CUDA/NCCL context memory, allocator fragmentation, and framework
overhead. Those exclusions make the unsharded model even less feasible on a
40 GB GPU.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Quadratic attention compute | 6,604.570 of 6,761.273 forward TFLOPs; scales as `l^2` | Context/sequence tokens (`cp`) and attention heads (`tp`) |
| Saved activations | 218.438 GB versus 9.089 GB for all unsharded training state | Context/sequence tokens (`cp`); TP over heads and MLP width |
| Position embedding | 265.422M parameters, 46.7% of all parameters | Sequence/token dimension (`cp`) |
| Global batch | Fixed global batch is 1 | `dp` must remain explicit but equal to 1 |

The primary parallelization target is therefore the token dimension, composed
with head/hidden tensor parallelism. Pure DP cannot fit the baseline and cannot
use more than one replica while preserving `global_batch_size=1`.
