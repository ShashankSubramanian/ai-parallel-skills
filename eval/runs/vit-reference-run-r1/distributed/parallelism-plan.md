# ViT Parallelism Plan

This plan is based on `distributed/analysis.md` and the unchanged representative
configuration. Implementation must not begin until the user explicitly
approves this plan.

## Constraints and Selected Strategy

| Field | Value |
| --- | --- |
| Representative model configuration | ViT, image `71 x 720 x 1440`, patch `2`, `l=259201`, `e=1024`, 24 layers, 16 heads, MLP `4096` |
| Memory target per device | 40 GB A100; leave material headroom for unmodeled CUDA/NCCL/framework memory |
| Global batch, microbatch, accumulation, inputs | `1 = 1 * dp(1) * 1`; one fixed image `(1,71,720,1440)` |
| Primary available devices | 32 GPUs: 8 nodes x 4 NVLink-connected GPUs |
| Additional requested scale | 64 and 128 GPUs where the selected mesh is valid; assumed 4 GPUs per node |
| Objective | Preserve global batch 1 and prioritize training throughput |
| Bottleneck | `l^2` attention compute and sequence-length saved activations; unsharded first-order memory is 227.527 GB |
| Primary candidate mesh | `dp=1, tp=4, cp=8` |
| Data-parallel wrapper | DDP interface, no-op at `dp=1`; DDP is selected because effective non-DP `reduce=["cp"]` metadata is required |
| Sequence parallelism | `false`; context parallelism already shards tokens, and TP sequence transitions are not needed to fit |
| Rank order | `tp-cp-dp`, fastest to slowest |

The configured global batch is one, so no valid plan can set `dp>1` while
preserving the batch equation. Extra GPUs therefore increase context
parallelism. Four-way TP maps each group to one NVLink-connected node and
shards all 16 attention heads evenly. CP groups connect the same local-GPU
position across nodes.

## Sequence Layout and Exact Semantics

The real sequence length `259201` is not divisible by the supported CP degrees.
The distributed path will append 31 dummy token positions, giving
`l_pad=259232`, which is divisible by 4, 8, 16, and 32. A key mask excludes all
dummy positions from attention. Real query outputs retain baseline order, and
dummy query outputs are discarded. Dropout is zero, so padding does not alter
the random-number contract.

The position embedding is sharded over CP after zero-padding its sequence
dimension. Comparisons, checkpoints, and gradient-equivalence tests include
only the first 259201 real positions; padded position slots are excluded from
the baseline mapping and must retain zero gradients.

CP rank 0 owns the CLS position. It computes the loss; other CP ranks use a
graph-connected zero scalar so every rank participates in the same backward
collectives. Detached logits/metrics are broadcast from CP rank 0 for
reporting.

## Sharded Operation and Communication Plan

Here `s=l_pad/cp`, `h=a/tp=4`, and each TP rank owns `e/tp=256` attention
channels. Inputs are replicated across TP and token-sharded across CP.

| Op group | Mesh | Logical | Local/sharded contract | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- | --- | --- |
| Patch embedding and token slice | `dp x tp x cp` | `b 71 720 1440 -> b p e` | Replicated convolution, then each CP rank keeps its `s` token interval | none | none | all-reduce `cp` |
| Position add / norms / residuals | `dp x tp x cp` | `b l e` | `b s e`, replicated over TP | none | none | all-reduce `cp` for replicated affine parameters |
| Attention QKV | `dp x tp x cp` | `b l e, e 3a d -> b l 3a d` | `b s e, e 3(a/tp)d -> b s 3(a/tp)d` | none | all-reduce `tp` | all-reduce `cp` |
| CP K/V exchange | `dp x tp x cp` | local Q, global K/V | Q `b h s d`; K/V gathered to `b h l_pad d` | all-gather K and V over `cp` | reduce-scatter K/V gradients over `cp` | none |
| Attention QK/PV | `dp x tp x cp` | `b a l d` attention | local queries `b h s d` against global K/V `b h l_pad d` | none beyond K/V exchange | none beyond K/V dual | none |
| Attention output projection | `dp x tp x cp` | `b l e, e e -> b l e` | `b s e/tp, e/tp e -> b s e` partial | all-reduce `tp`; add bias afterward | none | all-reduce `cp` |
| MLP first projection | `dp x tp x cp` | `b l e, e f -> b l f` | `b s e, e f/tp -> b s f/tp` | none | all-reduce `tp` | all-reduce `cp` |
| GELU | `dp x tp x cp` | `b l f -> b l f` | `b s f/tp` | none | none | none |
| MLP second projection | `dp x tp x cp` | `b l f, f e -> b l e` | `b s f/tp, f/tp e -> b s e` partial | all-reduce `tp`; add bias afterward | none | all-reduce `cp` |
| Final norm / classifier | `dp x tp x cp` | CLS `b e -> b 10` | CLS exists on CP rank 0; model state remains replicated over TP/CP | detached metric broadcast over `cp` | graph-connected zero path on non-owner CP ranks | all-reduce `cp` |

The K/V all-gather is chosen over a custom ring-attention kernel because it
keeps the baseline SDPA semantics and gives a tractable correctness path. It
still shards queries and TP heads. Its cost is approximately 0.265 GB of BF16
K/V payload per layer per TP rank at the primary shape, exchanged over each CP
group. This communication and the availability of a fused SDPA kernel for
`q_len=32404`, `kv_len=259232`, and head size 64 are primary performance risks.

## Parameter Metadata Handoff

Storage shapes below follow `named_parameters()` layout. `sharded`, `shared`,
and `reduce` are the exact planned `comm_metadata` values. DP gradient
synchronization is owned by DDP and is not listed in `reduce`.

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_embed.weight` | convolution weight | same | `(1024,71,2,2)` | same | `[None,None,None,None]` | `["tp","cp"]` | `["cp"]` | Replicated stem; CP ranks contribute token-slice partial wgrads; TP copies are duplicate |
| `patch_embed.bias` | output channel | same | `(1024,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Same stem ownership |
| `cls_token` | `1 1 e` | same | `(1,1,1024)` | same | `[None,None,None]` | `["tp","cp"]` | `["cp"]` | CLS is used only by CP rank 0; TP copies receive duplicate gradients |
| `pos_embed` | `1 l e` | `1 s e` | `(1,259201,1024)` | `(1,259232/cp,1024)` | `[None,"cp",None]` | `["tp"]` | `[]` | Distinct CP token shards; TP copies are duplicate; padded slots are masked |
| `blocks.*.norm{1,2}.{weight,bias}` | `e` affine | same | `(1024,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Replicated affine applied to CP-local tokens |
| `blocks.*.attn.qkv.weight` | `e 3e` | `e 3e/tp` | `(3072,1024)` | `(3072/tp,1024)` | `["tp",None]` | `["cp"]` | `["cp"]` | Column parallel over QKV/head output; CP contracts over token shards in wgrad |
| `blocks.*.attn.qkv.bias` | `3e` | `3e/tp` | `(3072,)` | `(3072/tp,)` | `["tp"]` | `["cp"]` | `["cp"]` | Matches QKV output shard |
| `blocks.*.attn.proj.weight` | `e e` | `e/tp e` | `(1024,1024)` | `(1024,1024/tp)` | `[None,"tp"]` | `["cp"]` | `["cp"]` | Row parallel over contracting head channels; local weight shard is distinct over TP |
| `blocks.*.attn.proj.bias` | `e` | same | `(1024,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Bias is added after TP reduction; CP token shards give partial wgrad |
| `blocks.*.mlp.net.0.weight` | `e f` | `e f/tp` | `(4096,1024)` | `(4096/tp,1024)` | `["tp",None]` | `["cp"]` | `["cp"]` | Column parallel over MLP output |
| `blocks.*.mlp.net.0.bias` | `f` | `f/tp` | `(4096,)` | `(4096/tp,)` | `["tp"]` | `["cp"]` | `["cp"]` | Matches first-projection output shard |
| `blocks.*.mlp.net.3.weight` | `f e` | `f/tp e` | `(1024,4096)` | `(1024,4096/tp)` | `[None,"tp"]` | `["cp"]` | `["cp"]` | Row parallel over contracting MLP channels |
| `blocks.*.mlp.net.3.bias` | `e` | same | `(1024,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Bias added after TP reduction |
| `norm.{weight,bias}` | `e` affine | same | `(1024,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Only CLS contributes nonzero final-norm gradient, owned by CP rank 0 |
| `head.weight` | `e 10` | same | `(10,1024)` | same | `[None,None]` | `["tp","cp"]` | `["cp"]` | Small replicated classifier; CP rank 0 computes the real loss |
| `head.bias` | `10` | same | `(10,)` | same | `[None]` | `["tp","cp"]` | `["cp"]` | Same classifier ownership |

For every block weight, the CP wgrad rule follows directly from its einsum. For
example, QKV computes `e s, s 3e/tp -> e 3e/tp`, and MLP FC2 computes
`f/tp s, s e -> f/tp e`: `s` is a CP-sharded contracting label, ranks hold the
same local parameter shard, and their partial wgrads must be summed over `cp`.
TP-sharded free dimensions inherit distinct parameter shards and must not be
reduced over TP. Replicated TP affine/bias copies see duplicate, not partial,
gradients because TP dgrad/output mappings restore the full residual stream.

## Finalized Topology Handoff

| Field | Planned value |
| --- | --- |
| Primary world size | `32 = dp(1) * tp(4) * cp(8)` |
| Scale-out world sizes | `64 = 1*4*16`; `128 = 1*4*32` |
| Launch method | Scheduler-neutral `torchrun`, 4 processes per node |
| Mesh dimensions | Explicit `dp`, `tp`, `cp`; primary `dp=1,tp=4,cp=8` |
| Sequence parallelism | `false` |
| Rank order | `tp-cp-dp` |
| Group tokens to build | `dp`, `tp`, `cp`, `dp-tp`, `dp-cp` |
| Hardware topology | Each contiguous TP group is one 4-GPU NVLink node; CP groups span nodes at equal local rank |
| Divisibility / padding | `heads % tp == 0`, `dim % tp == 0`, `mlp_dim % tp == 0`; sequence padded from 259201 to 259232 and masked; `l_pad % cp == 0` |

The group-token set contains every atomic dimension and every DP-composite
candidate induced by `shared=["tp"]`, `shared=["cp"]`, or `reduce=["cp"]`.
There is no separate sequence-parallel group.

## Per-GPU Memory Projection

These decimal-GB estimates use FP32 local parameters/gradients, two FP32 AdamW
states, BF16 major activations, fused SDPA, TP=4, and CP K/V all-gather. The
saved-activation estimate includes gathered K/V retained for backward. The
temporary and communication columns are planning reserves, not measured peaks:
2.0 GB for kernels/framework temporaries and 0.75 GB for collective staging.
CUDA context, allocator fragmentation, NCCL implementation details, and the
actual selected SDPA workspace remain unverified.

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `dp=1,tp=4,cp=4` | 0.569 | 0.569 | 1.138 | 28.479 | 2.000 | 0.750 | **33.505 GB** |
| 32 | `dp=1,tp=4,cp=8` | 0.437 | 0.437 | 0.873 | 17.572 | 2.000 | 0.750 | **22.068 GB** |
| 64 | `dp=1,tp=4,cp=16` | 0.370 | 0.370 | 0.740 | 12.118 | 2.000 | 0.750 | **16.349 GB** |
| 128 | `dp=1,tp=4,cp=32` | 0.337 | 0.337 | 0.674 | 9.391 | 2.000 | 0.750 | **13.489 GB** |

The 16-GPU row has much less unmodeled headroom and is secondary. The required
primary 32-GPU configuration has about 17.9 GB of headroom relative to 40 GB
under these assumptions. Runtime results are unverified because the launch
scripts must not be executed here.

## Scale Policy

| Scale | Model/input configuration | Mesh | Reason for changing model-parallel dimensions |
| --- | --- | --- | --- |
| Primary, 32 GPUs | Representative | `dp=1,tp=4,cp=8` | TP fills each NVLink island; CP is large enough for comfortable memory headroom |
| 64 GPUs | Representative | `dp=1,tp=4,cp=16` | Batch cannot increase DP; more CP reduces local queries and attention compute |
| 128 GPUs | Representative | `dp=1,tp=4,cp=32` | Same TP/node mapping; CP scales across 32 nodes and divides padded sequence |

World sizes below 16 are rejected by the planned launcher for this fixed
configuration. World sizes 16, 32, 64, and 128 are accepted only with
`dp=1,tp=4,cp=world_size/4`; 32 is the primary supported target. No claim is
made for a different image, model, batch, precision, optimizer, checkpointing
policy, TP degree, or CP degree without a new analysis or projection.

## Throughput Rationale, Costs, and Risks

Pros:

- TP=4 keeps frequent TP reductions within NVLink-connected nodes.
- CP and TP jointly divide the dominant attention work; ideal counted
  forward/backward work per GPU is about 633.9 TFLOPs at 32 GPUs, 317.0 at 64,
  and 158.5 at 128 before communication and imbalance.
- CP shards the 265M-element position embedding and most saved activations.
- Global batch and model semantics are preserved.

Cons and risks:

- Every layer all-gathers K/V across nodes; CP scaling can become network-bound.
- Fused SDPA support for very long K/V must be confirmed on the target PyTorch,
  CUDA, and A100 stack. Falling back to dense math attention is infeasible.
- Memory numbers are analytical, not allocator peaks.
- The replicated patch convolution does redundant work on every TP/CP rank,
  although it is only 0.151 TFLOPs versus 6,761 TFLOPs total forward work.
- At batch 1, DP cannot be used as the normal throughput scale-out axis.
- More CP ranks reduce local query work but do not reduce each rank's gathered
  global K/V size, so 128-GPU scaling efficiency is uncertain.

## Alternatives Considered

- **Pure DP:** rejected because the model does not fit unsharded and
  `global_batch_size=1` requires `dp=1`.
- **FSDP:** rejected as the primary strategy because activations and attention
  compute dominate, while CP requires non-DP wgrad reductions. The first-order
  FSDP path is not valid under the required metadata contract.
- **TP without CP:** rejected because saved activations remain too large and
  attention sequence work is not divided enough.
- **CP without TP:** possible in principle but communicates four times more K/V
  per rank and fails to exploit each node's NVLink island for head/MLP work.
- **TP sequence parallelism:** left disabled because CP already shards every
  token-preserving region and the primary mesh fits with headroom.
- **Spatial-only sharding:** rejected because patch convolution is negligible
  and global self-attention still requires a sequence-wide strategy.
- **Activation checkpointing:** not selected because the fixed configuration
  explicitly disables it.
- **Custom ring attention:** potentially lowers gathered-K/V memory but carries
  substantially higher implementation and numerical-correctness risk. It can
  be considered later if measured all-gather scaling is inadequate.

## Planned Files and Boundaries

All implementation artifacts will remain under `distributed/`:

- `comm.py`: mesh construction and exact group-token API.
- `mappings.py`: autograd-safe TP and CP split/gather/reduce mappings.
- `model.py`: complete distributed ViT, padded token layout, TP linears, and
  CP attention.
- `train_distributed.py`: immutable-config loading, initialization, DDP
  integration, optimizer, owner-rank loss, and metrics.
- `test_distributed.py`: topology plus forward/input-gradient/weight-gradient
  equivalence tests against the immutable baseline.
- `launch.sh`: scheduler-neutral `test`, `train`, and `--help` interface with
  mesh validation.
- `implementation-report.md`: files, supported configurations, exact commands,
  unverified runtime results, assumptions, and remaining risks.

## Validation Plan

1. Unit-test rank-coordinate conversion and every required group membership.
2. Test group-size-one mapping behavior and TP/CP collective forward/backward
   duals.
3. On reduced deterministic dimensions, load a baseline state into distributed
   shards and compare gathered forward logits.
4. Compare input gradients and every real baseline parameter gradient,
   including CP-reduced replicated parameters and gathered TP/CP shards.
5. Verify padded position slots have zero gradients and masked padding does not
   alter real-token results.
6. Validate launch rejection for mismatched world size, `dp>1` with global
   batch 1, unsupported TP/CP values, and non-divisible dimensions.
7. Provide exact 32-, 64-, and 128-GPU `torchrun`/`launch.sh` commands, but do
   not execute them. Their results will be explicitly marked **unverified**.
