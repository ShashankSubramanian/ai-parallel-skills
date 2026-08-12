# Model analysis: fixed-resolution 70-channel ViT

## Model contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Baseline | `models/vit.py::VisionTransformer` |
| Entry point | `VisionTransformer.forward(x)` |
| Inputs | Image tensor `x: (1, 70, 720, 1440)` in NCHW order |
| Targets | One class index `(1,)` for cross-entropy training |
| Outputs | Logits `(1, 1000)` |
| Dtype | BF16 parameters and activations |
| Optimizer | AdamW with FP32 first/second moments and an FP32 master-weight copy |
| Checkpointing | None |
| Fixed dimensions | Patch `2x2`, grid `360x720`, 259,200 patches, 259,201 tokens including CLS, embedding 1024, 24 blocks, 16 heads, head width 64, MLP width 4096 |
| Dynamic dimensions | None; this implementation targets only `720x1440` inputs |
| Divisibility | `720 % 2 == 0`, `1440 % 2 == 0`, `1024 % 16 == 0`; planned TP size must divide 16, 1024, and 4096 |
| Baseline adaptation | The baseline constructor accepts only a square `image_size`. The distributed copy must support a fixed `(height, width)` without modifying the baseline source. |

The repository contains no existing loss, optimizer setup, training loop,
distributed launcher, or executable ViT tests. Cross-entropy and AdamW are
therefore explicit assumptions for the new distributed training path.

## Shape trace

Let `b=1`, `c=70`, `h=720`, `w=1440`, `p=2`, `e=1024`, `a=16`,
`d=64`, `f=4096`, `v=1000`, `n=24`, and
`l=(h/p)(w/p)+1=259,201`.

| Stage | Count | Input | Output | Backward contract |
| --- | ---: | --- | --- | --- |
| Patch convolution | 1 | `b c h w` | `b e 360 720` | input dgrad; weight and bias wgrads summed over batch/grid |
| Flatten patches | 1 | `b e 360 720` | `b 259200 e` | reshape/transpose inverse |
| CLS concat + position add | 1 | patches, `(1,1,e)`, `(1,l,e)` | `b l e` | split token dgrads; positional and CLS wgrads |
| LayerNorm 1 | 24 | `b l e` | `b l e` | input dgrad; scale/bias reductions over `b,l` |
| QKV projection | 24 | `b l e` | `b l 3 a d` | linear dgrad/wgrad/bias contract |
| Scaled dot-product attention | 24 | Q/K/V `b a l d` | context `b l a d` | Q/K/V dgrads from global non-causal attention |
| Output projection | 24 | `b l e` | `b l e` | linear dgrad/wgrad/bias contract |
| LayerNorm 2 | 24 | `b l e` | `b l e` | input dgrad; scale/bias reductions over `b,l` |
| MLP FC1 + GELU | 24 | `b l e` | `b l f` | GELU dgrad then linear dgrad/wgrad/bias |
| MLP FC2 | 24 | `b l f` | `b l e` | linear dgrad/wgrad/bias contract |
| Final norm + CLS select | 1 | `b l e` | `b e` | dgrad only at the CLS sequence position |
| Classification head | 1 | `b e` | `b v` | linear dgrad/wgrad/bias contract |

For a linear operation `x @ W + bias`, the logical contracts are:

```text
forward:  b l e, e f -> b l f
dgrad:    b l f, f e -> b l e
wgrad:    e (b*l), (b*l) f -> e f
bias grad: sum over b,l
```

Attention is:

```text
qkv:    b l e, e 3a d -> b l 3a d
scores: b a l d, b a m d -> b a l m
ctx:    b a l m, b a m d -> b l a d
out:    b l a d, a d e -> b l e
```

## Operation groups and forward costs

Values are decimal GB and TFLOPs. Weight entries include biases. FLOPs count
multiply-add as two operations and cover forward only. Backward is estimated as
approximately twice forward for matmuls and attention.

| Op | Count | Contract | Weights (GB, BF16) | Cached activations (GB, BF16) | Forward FLOPs (TFLOPs) |
| --- | ---: | --- | ---: | ---: | ---: |
| `patch_embed` | 1 | `conv2d(b c h w, e c p p) -> b e h/p w/p` | 0.000575 | 0.145152 | 0.148636 |
| `pos_embed`, `cls_token` | 1 | `concat/add -> b l e` | 0.530846 | included downstream | negligible |
| `blocks.*.norm1/norm2` | 48 | `layer_norm(b l e) -> b l e` | 0.000197 | 25.480495 | small vs matmuls |
| `blocks.*.attn.qkv` | 24 | `b l e, e 3e -> b l 3e` | 0.151142 | 12.740248 | 39.138040 |
| `blocks.*.attn.sdpa` | 24 | `Q,K,V -> b l e` | 0 | 51.359123 | 6604.569811 |
| `blocks.*.attn.proj` | 24 | `b l e, e e -> b l e` | 0.050381 | 12.740248 | 13.046013 |
| `blocks.*.mlp.fc1` | 24 | `b l e, e f -> b l f` | 0.201523 | 12.740248 | 52.184054 |
| `blocks.*.mlp.gelu` | 24 | `gelu(b l f) -> b l f` | 0 | 50.960990 | small vs matmuls |
| `blocks.*.mlp.fc2` | 24 | `b l f, f e -> b l e` | 0.201376 | 50.960990 | 52.184054 |
| `norm` | 1 | `layer_norm(b l e) -> b l e` | 0.000004 | 0.530844 | small |
| `head` | 1 | `b e, e v -> b v` | 0.002050 | 0.000002 | 0.000002 |
| **Total** |  |  | **1.138094** | **about 217.66** | **6761.27** |

The saved-activation total is a conservative additive autograd-save estimate.
Some adjacent operations retain views of the same storage, so measured peak can
be lower. Conversely, the selected SDPA backend, temporary workspaces, allocator
fragmentation, and communication buffers can raise actual peak memory. The
estimate assumes a memory-efficient fused SDPA backend; materializing the full
`(b,a,l,l)` score matrix would require roughly 2.15 TB in BF16 per block and is
not viable.

## Parameter count

| Parameter group | Elements | BF16 memory (GB) |
| --- | ---: | ---: |
| Patch embedding | 287,744 | 0.000575 |
| CLS token | 1,024 | 0.000002 |
| Positional embedding | 265,421,824 | 0.530844 |
| 24 transformer blocks | 302,309,376 | 0.604619 |
| Final norm | 2,048 | 0.000004 |
| Classification head | 1,025,000 | 0.002050 |
| **Total** | **569,047,016** | **1.138094** |

The fixed learned positional table is 46.6% of all parameters. Transformer
weights are still important, but uncheckpointed sequence activations dominate
training memory.

## Memory summary before sharding

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 1.138 | `l*e`, `n*e*f` | BF16 |
| Gradients | 1.138 | same as weights | One BF16 gradient per parameter |
| AdamW moments | 4.552 | parameter count | Two FP32 states |
| FP32 master weights | 2.276 | parameter count | Conservative mixed-precision assumption |
| Saved activations | about 217.66 | `n*l*f`, `n*l*e`, attention Q/K/V | No checkpointing; fused SDPA required |
| Input | 0.145 | `b*c*h*w` | BF16 |
| Temporary/runtime | backend-dependent | `l`, local attention workspace | Not included in subtotal |
| **Persistent + saved subtotal** | **about 226.77** | sequence length | Does not fit one 40GB A100 |

Forward compute is about 6.76 PFLOPs; forward plus backward is approximately
20.28 PFLOPs per training step before optimizer work. Global attention accounts
for about 97.7% of forward FLOPs. Parallelism distributes this compute but does
not reduce the total arithmetic.

## Bottlenecks

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Saved sequence activations | About 217.66GB versus 1.14GB BF16 weights; scales with `n*l*e` and `n*l*f` | Context shard `l`; tensor shard attention heads and MLP `f` |
| Global attention compute | About 6.60 PFLOPs of 6.76 PFLOPs forward; scales as `l^2*e` | Context shard query tokens and tensor shard heads |
| Positional parameter table | 265.4M elements / 0.531GB BF16 | Context shard sequence dimension |
| Inter-node communication | Every local query needs global K/V for exact attention | CP K/V all-gather or ring attention; prefer exact padded-transport/all-gather first |

## Analysis conclusion

Data parallelism cannot help at batch size one and would replicate the memory
bottleneck. A composed context/tensor mesh is required. The hardware naturally
supports `tp=4` within each NVLink node and `cp=8` across nodes. CP partitions
the 259,201-token sequence; TP partitions 16 attention heads and the 4096-wide
MLP. The detailed collective, metadata, memory, and test handoff is in
`parallelism-plan.md`.
