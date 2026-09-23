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
  --master-addr HOST      Rendezvous host (default: first Slurm node)
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

Train mode requires dp=1 because the fixed global batch size is one. Other TP
and CP degrees are permitted when their product matches world size and the
model dimensions are divisible by TP. Unanalyzed meshes may run out of memory.

Run this command once inside a Slurm allocation. It starts one torchrun agent
per node through srun and Shifter.
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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
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
  if (( DP != 1 )); then
    echo "error: train mode requires dp=1 for global_batch_size=1" >&2
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
export NVIDIA_TF32_OVERRIDE=0
if (( REALISTIC_SHAPE == 1 )); then
  export RUN_REALISTIC_SHAPE=1
fi

TORCHRUN=(
  srun
  --nodes="$NNODES"
  --ntasks-per-node=1
  --gpus-per-node=4
  --cpus-per-task=128
  shifter
  --image=registry.nersc.gov/dasrepo/shas1693/sciml-pytorch:26.07
  torchrun
  --nnodes="$NNODES"
  --nproc-per-node=4
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
