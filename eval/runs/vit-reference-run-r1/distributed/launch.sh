#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  distributed/launch.sh test [OPTIONS]
  distributed/launch.sh train [OPTIONS]
  distributed/launch.sh --help

Options:
  --nnodes N              Number of nodes (default: 8)
  --nproc-per-node N      Processes/GPUs per node (default: 4)
  --node-rank N           This node's rank, 0..nnodes-1 (default: 0)
  --master-addr HOST      Rendezvous host (default: 127.0.0.1)
  --master-port PORT      Rendezvous port (default: 29500)
  --backend NAME          nccl or gloo (default: nccl)
  --dp N                  Data-parallel degree (default: 1)
  --tp N                  Tensor-parallel degree (default: 4)
  --cp N                  Context-parallel degree (default: 8)
  --config PATH           Training configuration (default: ../config.yaml)
  --realistic-shape       Also enable the non-compute realistic layout test
  --help                  Print this message

The mesh must satisfy:
  nnodes * nproc-per-node = dp * tp * cp

Train mode supports the approved fixed meshes:
  16 GPUs:  dp=1 tp=4 cp=4   (4 nodes x 4 GPUs)
  32 GPUs:  dp=1 tp=4 cp=8   (8 nodes x 4 GPUs; primary)
  64 GPUs:  dp=1 tp=4 cp=16  (16 nodes x 4 GPUs)
  128 GPUs: dp=1 tp=4 cp=32  (32 nodes x 4 GPUs)

Run this same command on every node, changing only --node-rank. This script
does not assume Slurm, PBS, or another scheduler.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

MODE=$1
shift
if [[ "$MODE" != "test" && "$MODE" != "train" ]]; then
  echo "error: mode must be 'test' or 'train', got '$MODE'" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR=$(dirname -- "$SCRIPT_DIR")
NNODES=8
NPROC_PER_NODE=4
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
BACKEND=nccl
DP=1
TP=4
CP=8
CONFIG="$TARGET_DIR/config.yaml"
REALISTIC_SHAPE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes) NNODES=$2; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE=$2; shift 2 ;;
    --node-rank) NODE_RANK=$2; shift 2 ;;
    --master-addr) MASTER_ADDR=$2; shift 2 ;;
    --master-port) MASTER_PORT=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --dp) DP=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --cp) CP=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --realistic-shape) REALISTIC_SHAPE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in NNODES NPROC_PER_NODE DP TP CP; do
  value=${!value_name}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $value_name must be a positive integer, got '$value'" >&2
    exit 2
  fi
done
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "error: NODE_RANK must be in [0, $((NNODES - 1))]" >&2
  exit 2
fi
if [[ -z "$MASTER_ADDR" ]]; then
  echo "error: MASTER_ADDR must not be empty" >&2
  exit 2
fi
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || \
   (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "error: MASTER_PORT must be in [1, 65535], got '$MASTER_PORT'" >&2
  exit 2
fi
if [[ "$BACKEND" != "nccl" && "$BACKEND" != "gloo" ]]; then
  echo "error: backend must be nccl or gloo, got '$BACKEND'" >&2
  exit 2
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MESH_SIZE=$((DP * TP * CP))
if (( WORLD_SIZE != MESH_SIZE )); then
  echo "error: world size $WORLD_SIZE != mesh product $DP*$TP*$CP=$MESH_SIZE" >&2
  exit 2
fi

if [[ "$MODE" == "train" ]]; then
  if (( DP != 1 || TP != 4 )); then
    echo "error: train mode requires dp=1 and tp=4" >&2
    exit 2
  fi
  if (( CP != 4 && CP != 8 && CP != 16 && CP != 32 )); then
    echo "error: train mode supports cp in {4,8,16,32}" >&2
    exit 2
  fi
  if (( NPROC_PER_NODE != 4 || NNODES != CP )); then
    echo "error: train mode requires 4 processes per node and nnodes=cp" >&2
    exit 2
  fi
  if [[ ! -f "$CONFIG" ]]; then
    echo "error: config file not found: $CONFIG" >&2
    exit 2
  fi
fi

export BACKEND
export DP_SIZE=$DP
export TP_SIZE=$TP
export CP_SIZE=$CP
if (( REALISTIC_SHAPE == 1 )); then
  export RUN_REALISTIC_SHAPE=1
fi

TORCHRUN=(
  torchrun
  --nnodes="$NNODES"
  --nproc-per-node="$NPROC_PER_NODE"
  --node-rank="$NODE_RANK"
  --rdzv-backend=c10d
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
)

cd "$TARGET_DIR"
if [[ "$MODE" == "test" ]]; then
  exec "${TORCHRUN[@]}" -m unittest -v distributed.test_distributed
else
  exec "${TORCHRUN[@]}" -m distributed.train_distributed \
    --config "$CONFIG" --backend "$BACKEND" --dp "$DP" --tp "$TP" --cp "$CP"
fi
