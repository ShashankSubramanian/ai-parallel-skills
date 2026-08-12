"""Minimal launchable BF16 training path for the distributed ViT."""

from __future__ import annotations

import argparse
import os

import torch
from torch.nn import functional as F

from . import comm
from .distributed.mappings import register_model_parallel_gradient_hooks
from .metadata import (
    init_params_for_shared_weights,
    validate_param_comm_metadata,
)
from .model import VisionTransformer
from .optimizer import FP32MasterAdamW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--channels", type=int, default=70)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=333)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the production BF16 training path requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    comm.initialize(tp_size=args.tp_size, cp_size=args.cp_size)

    # TP shards use distinct random streams; CP replicas of those shards use
    # the same stream and are synchronized again from metadata.
    torch.manual_seed(args.seed + comm.get_rank("tp"))
    torch.cuda.manual_seed(args.seed + comm.get_rank("tp"))
    model = VisionTransformer(
        image_size=(args.height, args.width),
        patch_size=args.patch_size,
        in_channels=args.channels,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device=device, dtype=torch.bfloat16)

    # Positional shards are not shared over CP and need distinct streams.
    with torch.random.fork_rng(devices=[local_rank]):
        positional_seed = args.seed + 10_000 + comm.get_rank("cp")
        torch.manual_seed(positional_seed)
        torch.cuda.manual_seed(positional_seed)
        torch.nn.init.trunc_normal_(model.pos_embed, std=0.02)

    validate_param_comm_metadata(model)
    init_params_for_shared_weights(model)
    hook_handles = register_model_parallel_gradient_hooks(model)
    optimizer = FP32MasterAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Every rank starts from the same logical sample and target. The model owns
    # the differentiable CP input split.
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 1)
    image = torch.randn(
        1,
        args.channels,
        args.height,
        args.width,
        device=device,
        dtype=torch.bfloat16,
    )
    target = torch.randint(
        args.num_classes, (1,), device=device, dtype=torch.long
    )

    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = F.cross_entropy(logits.float(), target, reduction="mean")
        scaled_loss = model.scale_loss_for_context_parallel(loss)
        scaled_loss.backward()
        optimizer.step()
        if comm.get_world_rank() == 0:
            allocated = torch.cuda.max_memory_allocated(device) / 1e9
            reserved = torch.cuda.max_memory_reserved(device) / 1e9
            print(
                f"step={step} loss={loss.item():.6f} "
                f"allocated_gb={allocated:.3f} reserved_gb={reserved:.3f}",
                flush=True,
            )

    for handle in hook_handles:
        handle.remove()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    comm.destroy()


if __name__ == "__main__":
    main()
