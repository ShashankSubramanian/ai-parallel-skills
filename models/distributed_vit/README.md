# Distributed ViT

This directory contains the `tp=4, cp=8` implementation for fixed
`(B,C,H,W)=(1,70,720,1440)` inputs. The original `models/vit.py` remains the
single-device oracle.

## Production launch

Launch one four-GPU TP group per node and CP across eight nodes:

```bash
torchrun \
  --nnodes=8 \
  --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m models.distributed_vit.train \
  --tp-size 4 \
  --cp-size 8 \
  --steps 1
```

The launcher uses BF16 model shards and `FP32MasterAdamW`, which keeps FP32
master parameters plus FP32 Adam moments. Each rank starts from the same
logical training sample. Replace the synthetic sample in `train.py` with the
application data loader while preserving the CP loss scaling.

## Correctness tests

World-size-one behavior:

```bash
python -m unittest \
  models.distributed_vit.tests.test_rank_generator \
  models.distributed_vit.tests.test_metadata \
  models.distributed_vit.tests.test_equivalence
```

Combined TP/CP equivalence on four CPU processes:

```bash
TP_SIZE=2 CP_SIZE=2 BACKEND=gloo \
torchrun --standalone --nproc-per-node=4 \
  -m unittest models.distributed_vit.tests.test_equivalence
```

Use `BACKEND=nccl` on four GPUs. The equivalence suite copies the immutable
baseline's full parameters into exact local shards, then compares forward
logits, the reconstructed full input gradient, and every named logical
parameter gradient. Its `8x12` image creates uneven CP sequence lengths because
only CP rank zero owns CLS.

The full production-shape smoke test is intentionally opt-in:

```bash
RUN_PRODUCTION_SHAPE=1 TP_SIZE=4 CP_SIZE=8 BACKEND=nccl \
torchrun \
  --nnodes=8 \
  --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m unittest models.distributed_vit.tests.test_equivalence
```

It checks finite forward/backward results and asserts that each rank reserves
less than 40 decimal GB. The exact peak depends on the installed PyTorch,
CUDA, NCCL, and SDPA backend.

## Runtime requirements

- PyTorch with distributed support.
- NCCL and CUDA-capable PyTorch for production.
- A memory-efficient SDPA backend; a backend that materializes the full
  attention score matrix cannot run the production shape.
