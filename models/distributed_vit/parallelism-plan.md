# Parallelism plan: fixed-resolution 70-channel ViT

## Constraints and selected mesh

| Field | Value |
| --- | --- |
| Memory target per device | 40GB A100, with at least 8GB runtime/allocator safety margin |
| Global batch and representative input | Batch 1, BF16 `(1,70,720,1440)`, class target `(1,)` |
| Model | Patch `2x2`, 259,201 tokens, embedding 1024, 24 blocks, 16 heads, MLP 4096, 1000 classes |
| Optimizer | AdamW, FP32 moments and FP32 master weights |
| Activation checkpointing | None |
| Available devices | 32 A100 40GB |
| Hardware | 8 nodes, 4 NVLink-connected GPUs per node; inter-node fabric bandwidth unspecified |
| Bottleneck | Uncheckpointed sequence activations (about 217.66GB) and quadratic global attention |
| Selected mesh | `tp=4, cp=8`, no data-parallel axis |
| Rank order | `tp-cp` |
| Rationale | Keep each four-rank TP group within a node; use all eight nodes to partition tokens |

Batch size one prevents useful data parallelism. Pipeline parallelism is out of
scope. The selected plan composes:

- tensor parallelism over four local GPUs for attention heads, QKV/projection
  features, patch output channels, and MLP hidden features;
- context parallelism over eight nodes for image patches, positional
  parameters, sequence activations, and attention queries.

## Logical and local sequence layout

The image produces a `360x720` patch grid, or 259,200 patch tokens. CP uses
contiguous horizontal patch-grid slabs:

```text
input per CP rank:  (1, 70, 90, 1440)
patch grid per CP rank: (45, 720)
patches per CP rank: 32,400
```

CP rank 0 prepends the CLS token and has logical sequence length 32,401. Other
CP ranks have logical sequence length 32,400. Positional parameters are sharded
to match those logical ranges.

NCCL all-gather requires equal transport shapes. For K/V transport only, ranks
with 32,400 tokens append one zero communication slot. All ranks gather buffers
of length 32,401, then strip each source rank back to its logical length before
concatenating K/V. SDPA therefore sees exactly 259,201 keys and values and no
padding token. This preserves baseline attention semantics without allocating a
quadratic attention mask.

## Operation and collective plan

Here `l_i` is 32,401 on CP rank 0 and 32,400 elsewhere, `m=259,201`,
`a/tp=4`, `e/tp=256`, and `f/tp=1024`.

| Op group | Local contract | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- |
| Input/patch split | `b c h w -> b c h/cp w` | split/scatter over CP at input boundary | gather only for tests or input-gradient reconstruction | none |
| `patch_embed` | `conv2d(b c h/cp w, e/tp c p p) -> b e/tp 45 720` | all-gather output channels over TP once | all-reduce patch-input dgrad over TP | all-reduce over CP |
| CLS/position add | local `b l_i e + 1 l_i e` | none; CLS exists only on CP rank 0 | none | positional table is CP-sharded; CLS wgrad all-reduce over CP |
| LayerNorm | `layer_norm(b l_i e)` | none | none | all-reduce scale/bias wgrads over CP |
| QKV projection | `b l_i e, e 3(a/tp)d -> b l_i 3(a/tp)d` | none | all-reduce activation dgrad over TP | all-reduce over CP |
| K/V globalization | local `b a/tp l_i d -> b a/tp m d` | padded all-gather and exact unpadding over CP | pad-aware reduce-scatter/split over CP | none |
| SDPA | `Q: b a/tp l_i d`, `K,V: b a/tp m d -> b l_i (a/tp)d` | none after K/V gather | handled through K/V mapping | none |
| Attention output projection | `b l_i e/tp, e/tp e -> b l_i e` | all-reduce partial output over TP | none beyond paired TP mapping | all-reduce over CP |
| MLP FC1 | `b l_i e, e f/tp -> b l_i f/tp` | none | all-reduce activation dgrad over TP | all-reduce over CP |
| GELU | `b l_i f/tp -> b l_i f/tp` | none | none | none |
| MLP FC2 | `b l_i f/tp, f/tp e -> b l_i e` | all-reduce partial output over TP | none beyond paired TP mapping | all-reduce over CP |
| Final norm | `b l_i e -> b l_i e` | none | none | all-reduce scale/bias wgrads over CP |
| CLS handoff | CP rank 0 `b e ->` all CP ranks `b e` | broadcast over CP in each TP lane | reduce to CP rank 0 | none |
| Head/loss | `b e, e v -> b v`; each CP rank uses loss divided by `cp` | logits already replicated; scalar may be checked by all-reduce | CP broadcast backward returns summed CLS dgrad to owner | all-reduce head wgrads over CP |

The differentiable mapping pairs are:

| Forward | Backward |
| --- | --- |
| TP identity/copy into a column-parallel op | TP all-reduce |
| TP all-reduce after a row-parallel op | identity |
| TP all-gather patch output channels | TP reduce-scatter |
| CP padded K/V all-gather + unpad | pad-aware CP split/reduce-scatter |
| CP CLS broadcast | CP reduce to owner |

Collectives are no-ops when their group size is one. Every rank in a group must
invoke collectives in identical order even though CP logical sequence lengths
differ.

## Parameter metadata handoff

Metadata follows stored PyTorch layouts. `Linear.weight` is `(out, in)`;
`Conv2d.weight` is `(out, in, kh, kw)`.

| Parameter pattern | Full shape | Local shape on `tp=4,cp=8` | `sharded` | `shared` | `reduce` | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| `patch_embed.weight` | `(1024,70,2,2)` | `(256,70,2,2)` | `["tp",null,null,null]` | `["cp"]` | `["cp"]` | Output-channel TP; replicated across image slabs |
| `patch_embed.bias` | `(1024,)` | `(256,)` | `["tp"]` | `["cp"]` | `["cp"]` | Matches local output channels |
| `cls_token` | `(1,1,1024)` | same | `[null,null,null]` | `["tp","cp"]` | `["cp"]` | Replicated storage; only CP rank 0 inserts it |
| `pos_embed` | `(1,259201,1024)` | rank 0 `(1,32401,1024)`, others `(1,32400,1024)` | `[null,"cp",null]` | `["tp"]` | `[]` | Uneven CP sequence shard, replicated over TP |
| `blocks.*.norm1/2.weight,bias` | `(1024,)` | same | `[null]` | `["tp","cp"]` | `["cp"]` | Replicated feature normalization, partial CP-token wgrads |
| `blocks.*.attn.qkv.weight` | reshaped `(3,16,64,1024)` | `(3,4,64,1024)` | `[null,"tp",null,null]` | `["cp"]` | `["cp"]` | Explicit stored head axis avoids block-cyclic fused-QKV metadata |
| `blocks.*.attn.qkv.bias` | reshaped `(3,16,64)` | `(3,4,64)` | `[null,"tp",null]` | `["cp"]` | `["cp"]` | Matches the explicit QKV head shard |
| `blocks.*.attn.proj.weight` | `(1024,1024)` | `(1024,256)` | `[null,"tp"]` | `["cp"]` | `["cp"]` | Row-parallel attention projection |
| `blocks.*.attn.proj.bias` | `(1024,)` | same | `[null]` | `["tp","cp"]` | `["cp"]` | Added after TP output reduction |
| `blocks.*.mlp.fc1.weight` | `(4096,1024)` | `(1024,1024)` | `["tp",null]` | `["cp"]` | `["cp"]` | Column-parallel MLP expansion |
| `blocks.*.mlp.fc1.bias` | `(4096,)` | `(1024,)` | `["tp"]` | `["cp"]` | `["cp"]` | Matches FC1 output shard |
| `blocks.*.mlp.fc2.weight` | `(1024,4096)` | `(1024,1024)` | `[null,"tp"]` | `["cp"]` | `["cp"]` | Row-parallel MLP contraction |
| `blocks.*.mlp.fc2.bias` | `(1024,)` | same | `[null]` | `["tp","cp"]` | `["cp"]` | Added after TP output reduction |
| `norm.weight,bias` | `(1024,)` | same | `[null]` | `["tp","cp"]` | `["cp"]` | Final norm sees a CP token shard |
| `head.weight` | `(1000,1024)` | same | `[null,null]` | `["tp","cp"]` | `["cp"]` | Small replicated head; CP-scaled duplicate loss |
| `head.bias` | `(1000,)` | same | `[null]` | `["tp","cp"]` | `["cp"]` | Same head policy |

Ordinary DP reduction is absent because `dp=1`. Each listed non-DP reduction
must run exactly once. Parameters with `shared` metadata are initialized on one
group root and broadcast so all replicas begin identically.

## Final topology handoff

| Field | Planned value |
| --- | --- |
| World size | 32 |
| Mesh dimensions | `tp=4, cp=8` |
| Rank order | `tp-cp` |
| TP groups | Contiguous ranks `[0..3]`, `[4..7]`, ..., `[28..31]` |
| CP groups | Same local TP coordinate across nodes: `[0,4,...,28]`, ..., `[3,7,...,31]` |
| Complete group-token set | `tp`, `cp` |
| Topology invariant | `world_size == tp * cp == 32` |
| Production launch | `torchrun --nnodes=8 --nproc-per-node=4 --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT -m models.distributed_vit.train ...` |
| Hardware mapping | One TP group per four-GPU NVLink node; CP spans nodes |
| Divisibility | `16 % tp == 0`, `1024 % tp == 0`, `4096 % tp == 0`, `360 % cp == 0` |
| Uneven rule | CLS makes rank 0 one token longer; pad only K/V communication buffers, then remove padding |

`comm.py` is the only module allowed to call `torch.distributed.new_group`.
Rank generation remains pure and side-effect free.

## Per-GPU memory projection

These decimal-GB estimates use the same conservative activation accounting as
`analysis.md`. Optimizer includes two FP32 moments and one FP32 master copy.
Temporary and communication values are explicit reserves, not measured peaks.
CUDA context, kernels, NCCL internals, and allocator fragmentation require an
additional 4-8GB operational reserve.

| GPUs | Mesh | Weights | Gradients | Optimizer + master | Saved activations | Temporary | Communication | Model-accounted total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `tp=1,cp=1` | 1.138 | 1.138 | 6.829 | 217.658 | 4.000 | 0 | 230.763 |
| 8 | `tp=1,cp=8` | 0.674 | 0.674 | 4.042 | 27.207 | 2.000 | 2.124 | 36.721 |
| 16 | `tp=2,cp=8` | 0.371 | 0.371 | 2.227 | 16.831 | 1.500 | 1.062 | 22.362 |
| 32 | `tp=4,cp=8` | 0.220 | 0.220 | 1.320 | 11.643 | 1.000 | 0.531 | 14.934 |

The eight-GPU configuration is diagnostic and risky on 40GB because runtime
overhead can exceed the remaining margin. The 16- and 32-GPU meshes are
expected to fit, with 32 GPUs providing the intended safety margin. Actual peak
memory must be measured with the production PyTorch/CUDA/SDPA stack.

## Implementation boundaries

All new files remain under `models/distributed_vit/`; `models/vit.py` remains
unchanged.

Planned modules:

```text
models/distributed_vit/
  __init__.py
  analysis.md
  parallelism-plan.md
  rank_generator.py
  comm.py
  metadata.py
  model.py
  train.py
  distributed/
    __init__.py
    helpers.py
    mappings.py
  tests/
    test_rank_generator.py
    test_metadata.py
    test_equivalence.py
```

The distributed model mirrors the baseline's `Attention`, `MLP`, `Block`, and
`VisionTransformer` boundaries while replacing only the operations required by
TP/CP.

## Validation plan

1. Pure rank-generator tests verify exact TP/CP memberships, orthogonality,
   coverage, and invalid world-size rejection.
2. Group-size-one tests verify all mappings are numerical no-ops.
3. Metadata tests require every trainable parameter to have a valid
   `comm_metadata` entry and check local storage shapes.
4. Baseline parameter-copy helpers slice each logical parameter according to
   metadata, including uneven positional shards.
5. Distributed equivalence tests compare:
   - reconstructed forward logits;
   - reconstructed full input gradients;
   - every named logical parameter gradient.
6. Routine tests use a small rectangular, divisible configuration on one and
   four local ranks. A production-shape smoke test uses all 32 GPUs and checks
   finite forward/backward values and peak allocated/reserved memory.
7. BF16 tolerances account for reduction-order differences. A small FP32 test
   is retained where the backend supports it to isolate semantic errors.

## Pros, cons, and risks

Pros:

- CP attacks the dominant sequence-activation memory directly.
- TP maps to the four NVLink GPUs per node and shards attention/MLP compute.
- Exact unpadding preserves the 259,201-token baseline semantics.
- No full attention score matrix is materialized.
- The 32-GPU modeled footprint leaves substantial room under 40GB.

Cons and risks:

- Every block globalizes K/V across eight nodes; performance depends strongly
  on inter-node bandwidth and overlap.
- Uneven logical CP shards require carefully ordered padded communication.
- BF16 and distributed reduction order will not be bitwise identical.
- PyTorch may select a non-memory-efficient SDPA backend for some shape/backend
  combinations; the launcher must verify the fused backend.
- A learned absolute positional table ties this implementation to the fixed
  resolution.
- No activation checkpointing means saved activations remain the largest local
  allocation even after sharding.

## Alternatives considered

- **Data parallelism:** rejected because batch size is one and activations do
  not fit on one device.
- **FSDP/ZeRO only:** rejected because persistent state is about 9.1GB while
  saved activations are about 217.66GB.
- **CP=32 without TP:** reduces sequence state but places all attention heads
  and MLP width on each GPU and increases context-group communication span.
- **TP=8, CP=4:** rejected because TP would cross node boundaries and CP would
  reduce the dominant sequence dimension less.
- **Spatial parallelism only:** patch convolution is local, but every
  transformer block immediately requires global attention; CP expresses the
  required sequence semantics directly.
- **Ring attention:** can reduce K/V gather-buffer memory, but is substantially
  more complex. The selected exact all-gather design fits the memory target and
  provides a simpler correctness baseline. Ring attention remains a later
  performance optimization.
- **Activation checkpointing:** would lower memory further but contradicts the
  selected no-checkpointing requirement.

## Approval gate

This plan and `analysis.md` complete the required analysis/planning phase.
Implementation of topology, parameter annotations, mappings, model code,
launcher, and equivalence tests starts only after explicit approval of these
two artifacts.
