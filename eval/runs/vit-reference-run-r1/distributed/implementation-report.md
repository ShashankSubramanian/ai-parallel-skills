# Distributed ViT Implementation Report

## Completion Status

The approved eager-mode distributed implementation is complete under
`distributed/`. The immutable baseline files `vit.py`, `config.yaml`, and
`train.py` were not modified.

Runtime tests and training were intentionally **not executed**. Their results
are **unverified** pending execution on the target PyTorch/CUDA/A100 system.
This development environment does not have PyTorch installed, so even local
runtime imports could not be checked.

## Implemented Strategy

- Primary mesh: `dp=1, tp=4, cp=8` on 32 GPUs.
- Scale-out meshes:
  - 64 GPUs: `dp=1, tp=4, cp=16`.
  - 128 GPUs: `dp=1, tp=4, cp=32`.
- Secondary supported mesh: 16 GPUs with `dp=1, tp=4, cp=4`.
- Rank order: `tp-cp-dp`.
- TP groups are contiguous four-rank groups intended to remain inside each
  four-GPU NVLink node.
- CP groups connect matching local-GPU positions across nodes.
- DDP remains the explicit DP wrapper; `dp=1` because global batch is fixed at
  one.
- Attention heads and MLP hidden features are TP-sharded.
- The sequence, position embedding, and query tokens are CP-sharded.
- K/V are gathered over CP for exact global attention; their gradients use the
  reduce-scatter dual.
- Real sequence length 259201 is padded to 259232. Gathered K/V are sliced back
  to the real length, so dummy positions cannot influence real tokens.
- QKV uses head-wise strided logical sharding while retaining local storage
  shape `(3e/tp, e)` and metadata `["tp", None]`. Copy/reconstruction tests
  explicitly restore baseline Q/K/V row order.
- Every trainable parameter has validated `comm_metadata`; CP reductions are
  consumed by the DDP communication hook.
- Sequence parallelism and activation checkpointing remain disabled.

## Files Produced

| File | Purpose |
| --- | --- |
| `analysis.md` | Fixed-configuration shapes, parameter count, memory, FLOPs, and bottlenecks |
| `parallelism-plan.md` | Approved mesh, collectives, metadata handoff, memory projection, and validation plan |
| `rank_generator.py` | Side-effect-free orthogonal hyperrectangle rank generation |
| `comm.py` | Device selection, distributed initialization, process groups, accessors, and teardown |
| `helpers.py` | Split utilities and non-mutating collective primitives |
| `mappings.py` | Autograd collective mappings and DDP gradient-reduction hook |
| `metadata.py` | Metadata attachment support, validation, and shared-parameter synchronization |
| `model.py` | Complete TP+CP ViT implementation |
| `train_distributed.py` | Distributed fixed-config training entry point |
| `test_distributed.py` | Topology, metadata, shared-init, forward, input-gradient, and named weight-gradient tests |
| `launch.sh` | Perlmutter Slurm/Shifter `test`, `train`, and `--help` interface |
| `implementation-report.md` | This completion and verification record |

## Launch Commands

Run each command once inside a Perlmutter Slurm allocation. The launcher starts
one `torchrun` agent per node through `srun` and Shifter. No command below has
been executed here.

### Help

```bash
./distributed/launch.sh --help
```

Result: **unverified** (the script was syntax-checked but not executed).

### Single-process correctness test

```bash
./distributed/launch.sh test \
  --nnodes 1 --nproc-per-node 1 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --backend gloo --dp 1 --tp 1 --cp 1
```

Result: **unverified**.

### Small combined TP+CP correctness test

```bash
./distributed/launch.sh test \
  --nnodes 1 --nproc-per-node 4 \
  --master-addr 127.0.0.1 --master-port 29500 \
  --backend nccl --dp 1 --tp 2 --cp 2
```

Result: **unverified**.

### Primary 32-GPU correctness test

```bash
./distributed/launch.sh test \
  --nnodes 8 --nproc-per-node 4 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --backend nccl --dp 1 --tp 4 --cp 8 --realistic-shape
```

Result: **unverified**.

The `--realistic-shape` option enables fixed-shape/divisibility contract
assertions. Numerical forward/backward equivalence still uses a small model so
the test is suitable for routine runs.

### Primary 32-GPU training

```bash
./distributed/launch.sh train \
  --nnodes 8 --nproc-per-node 4 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --backend nccl --dp 1 --tp 4 --cp 8 \
  --config config.yaml
```

Result: **unverified**.

### 64-GPU training

```bash
./distributed/launch.sh train \
  --nnodes 16 --nproc-per-node 4 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --backend nccl --dp 1 --tp 4 --cp 16 \
  --config config.yaml
```

Result: **unverified**.

### 128-GPU training

```bash
./distributed/launch.sh train \
  --nnodes 32 --nproc-per-node 4 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --backend nccl --dp 1 --tp 4 --cp 32 \
  --config config.yaml
```

Result: **unverified**.

## Validation Coverage

`test_distributed.py` implements:

- Known-rank rank-generator membership checks.
- Mesh product and group-accessor checks.
- Single-device behavior when launched with `dp=tp=cp=1`.
- Combined TP+CP behavior under a multi-rank launch.
- Metadata coverage and group-name validation for every trainable parameter.
- Shared-parameter synchronization checks.
- Explicit baseline-to-local parameter copying, including padded position
  shards and head-wise fused-QKV shards.
- FP32 forward-logit equivalence.
- FP32 input-gradient equivalence.
- Named gradient equivalence for every trainable parameter after reconstruction.
- Zero-gradient checks for padded position slots.
- Realistic fixed-shape and divisibility checks behind `--realistic-shape`.

The tests use `torch.testing.assert_close` with `rtol=2e-4` and `atol=2e-5`.
The tolerance allows distributed reduction-order and SDPA-kernel differences
while remaining tight for FP32 test execution.

## Checks Performed Here

| Check | Result |
| --- | --- |
| `bash -n distributed/launch.sh` | Passed |
| Python syntax compilation for all files under `distributed/` | Passed |
| IDE linter diagnostics for `distributed/` | No errors reported |
| `git diff --check` for generated artifacts | Passed |
| Python module import | Blocked: `ModuleNotFoundError: No module named 'torch'` |
| Distributed unit tests | Not run; **unverified** |
| 32/64/128-GPU training | Not run; **unverified** |

No generated launcher was executed.

## Supported and Rejected Configurations

Supported training configurations:

- The supplied fixed model, data, precision, optimizer, dropout, and
  checkpointing configuration.
- `dp=1` with positive TP and CP degrees whose product equals world size.
- TP degrees that divide the configured head count, hidden size, and MLP hidden
  size. CP padding is selected dynamically from the requested CP degree.
- The analyzed meshes remain `dp=1,tp=4,cp={4,8,16,32}`. Other valid meshes
  are experimental and may run out of memory.
- NCCL is the intended training backend. Gloo exists for CPU/small tests.

Rejected by validation:

- `dp>1`, because it violates the fixed global-batch equation.
- Mesh products that do not equal `WORLD_SIZE`.
- Non-positive parallel degrees.
- Nonzero dropout or enabled activation checkpointing.
- Hidden size, head count, or MLP width not divisible by TP.

Training meshes outside the analyzed set are accepted for experimentation but
are not claimed to fit in device memory. The test path uses smaller model
dimensions to exercise world-size-one and combined-axis correctness.

## Remaining Assumptions and Unverified Claims

1. The target environment provides a compatible PyTorch release with
   `torch.distributed`, DDP `GradBucket.parameters()/gradients()`, BF16 A100
   autocast, and SDPA.
2. The selected fused/memory-efficient SDPA backend supports local query length
   32404 and key/value length 259201 at the 32-GPU mesh. A dense math fallback
   is infeasible.
3. The target launcher permits all ranks to create the requested NCCL
   subgroups in deterministic order.
4. The target inter-node fabric can sustain the per-layer CP K/V all-gathers.
   Scaling efficiency, especially at CP=32, is unverified.
5. The analytical per-GPU memory estimates are not measured allocator peaks.
6. DDP bucket layouts are identical within each CP group, as required by the
   mixed-signature reduction hook.
7. Exact runtime numerical tolerances may need adjustment for the target SDPA
   kernel, but any adjustment must follow inspection of finite per-tensor
   diagnostics rather than hiding a systematic mismatch.
8. Checkpoint save/load conversion between full baseline tensors and sharded
   runtime tensors is not implemented; initialization and equivalence copying
   are implemented.
9. Fault tolerance, elastic membership, scheduler integration, and
   `torch.compile` are outside this implementation.
