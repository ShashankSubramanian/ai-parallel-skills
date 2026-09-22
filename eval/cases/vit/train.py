"""Single-device training path for the ViT evaluation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vit import VisionTransformer


class SyntheticImages(Dataset):
    """Generate deterministic samples lazily to avoid storing large images."""

    def __init__(
        self,
        *,
        sample_count: int,
        channels: int,
        image_size: tuple[int, int],
        class_count: int,
        seed: int,
    ) -> None:
        self.sample_count = sample_count
        self.channels = channels
        self.image_size = image_size
        self.class_count = class_count
        self.seed = seed

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(
            self.channels,
            *self.image_size,
            generator=generator,
        )
        target = torch.randint(
            self.class_count,
            (),
            generator=generator,
        )
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device or 'auto'.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    required = {"model", "data", "training"}
    if set(config) != required:
        raise ValueError(
            f"{path} must contain exactly {sorted(required)}, got {sorted(config)}"
        )
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(model_config: dict[str, Any]) -> VisionTransformer:
    image_size = tuple(int(value) for value in model_config["image_size"])
    return VisionTransformer(
        image_size=image_size,
        patch_size=int(model_config["patch_size"]),
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        dim=int(model_config["dim"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
    )


def build_dataloader(
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    image_size = tuple(int(value) for value in model_config["image_size"])
    dataset = SyntheticImages(
        sample_count=int(data_config["num_samples"]),
        channels=int(model_config["in_channels"]),
        image_size=image_size,
        class_count=int(model_config["num_classes"]),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config["global_batch_size"]),
        shuffle=bool(data_config["shuffle"]),
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    device = resolve_device(args.device)
    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(model_config).to(device)
    dataloader = build_dataloader(model_config, data_config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

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
                logits = model(images)
                loss = F.cross_entropy(logits.float(), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_examples += targets.numel()

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


if __name__ == "__main__":
    main()
