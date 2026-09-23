"""Distributed training entry point for the fixed ViT configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from train import build_dataloader, load_config

from . import comm
from .helpers import _reduce
from .mappings import init_ddp_model_and_reduction_hooks
from .metadata import init_params_for_shared_weights, validate_param_comm_metadata
from .model import DistributedVisionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--backend", default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--debug-topology", action="store_true")
    return parser.parse_args()


def validate_training_mesh(config: dict[str, Any], args: argparse.Namespace) -> None:
    if min(args.dp, args.tp, args.cp) < 1:
        raise ValueError("dp, tp, and cp must be positive")
    if args.dp != 1:
        raise ValueError(
            "global_batch_size=1 requires dp=1; changing dp would alter semantics"
        )
    expected_world = args.dp * args.tp * args.cp
    launched_world = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world != expected_world:
        raise ValueError(
            f"WORLD_SIZE={launched_world} but dp*tp*cp={expected_world}"
        )
    data = config["data"]
    if int(data["global_batch_size"]) != 1:
        raise ValueError("this implementation is fixed to global_batch_size=1")
    if int(data["num_samples"]) < 1:
        raise ValueError("num_samples must be positive")
    model = config["model"]
    if int(model["heads"]) % args.tp or int(model["dim"]) % args.tp:
        raise ValueError("heads and dim must be divisible by tp")
    hidden = int(int(model["dim"]) * float(model["mlp_ratio"]))
    if hidden % args.tp:
        raise ValueError("MLP hidden size must be divisible by tp")
    if float(model["dropout"]) != 0.0:
        raise ValueError("the approved distributed path requires dropout=0")
    if bool(config["training"]["activation_checkpointing"]):
        raise ValueError("the approved fixed configuration disables checkpointing")


def build_model(
    model_config: dict[str, Any], *, cp: int
) -> DistributedVisionTransformer:
    return DistributedVisionTransformer(
        image_size=tuple(int(value) for value in model_config["image_size"]),
        patch_size=int(model_config["patch_size"]),
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        dim=int(model_config["dim"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        sequence_padding_multiple=cp,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_training_mesh(config, args)
    dimensions = {"dp": args.dp, "tp": args.tp, "cp": args.cp}
    comm.initialize(
        dimensions=dimensions,
        backend=args.backend,
        debug=args.debug_topology,
    )
    try:
        device = torch.device(
            f"cuda:{comm.get_local_rank()}" if args.backend == "nccl" else "cpu"
        )
        training_config = config["training"]
        seed = int(training_config["seed"])
        torch.manual_seed(seed + comm.get_world_rank())
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + comm.get_world_rank())

        model = build_model(config["model"], cp=args.cp).to(device)
        validate_param_comm_metadata(model, print_summary=args.debug_topology)
        init_params_for_shared_weights(model)
        model = init_ddp_model_and_reduction_hooks(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        dataloader = build_dataloader(config["model"], config["data"], seed)

        for epoch in range(int(training_config["epochs"])):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=training_config["precision"] == "bfloat16",
                ):
                    local_logits = model(images)
                    if comm.get_rank("cp") == 0:
                        loss = F.cross_entropy(local_logits.float(), targets)
                    else:
                        loss = local_logits.float().sum() * 0.0
                loss.backward()
                optimizer.step()

                logits = _reduce(local_logits.detach(), "cp")
                reported_loss = _reduce(loss.detach(), "cp")
                if comm.get_world_rank() == 0:
                    total_loss += reported_loss.item() * targets.numel()
                    total_correct += (
                        (logits.argmax(dim=-1) == targets).sum().item()
                    )
                    total_examples += targets.numel()

            if comm.get_world_rank() == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "loss": total_loss / total_examples,
                            "accuracy": total_correct / total_examples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        comm.destroy()


if __name__ == "__main__":
    main()
