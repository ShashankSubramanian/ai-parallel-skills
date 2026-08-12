"""Parameter communication metadata and validation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn

from . import comm

CommMetadata = dict[str, list[str | None] | list[str]]


def set_comm_metadata(
    parameter: nn.Parameter,
    *,
    sharded: Sequence[str | None],
    shared: Sequence[str] = (),
    reduce: Sequence[str] = (),
) -> None:
    """Attach the approved communication contract to a parameter."""
    parameter.comm_metadata = {  # type: ignore[attr-defined]
        "sharded": list(sharded),
        "shared": list(shared),
        "reduce": list(reduce),
    }


def validate_param_comm_metadata(model: nn.Module) -> None:
    """Fail if a trainable parameter has incomplete or invalid metadata."""
    known_groups = set(comm.get_names())
    required_keys = {"sharded", "shared", "reduce"}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        metadata = getattr(parameter, "comm_metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError(f"trainable parameter {name!r} has no comm_metadata")
        if set(metadata) != required_keys:
            raise ValueError(
                f"parameter {name!r} metadata keys are {sorted(metadata)}, "
                f"expected {sorted(required_keys)}"
            )
        if not all(isinstance(metadata[key], list) for key in required_keys):
            raise TypeError(f"parameter {name!r} metadata values must be lists")
        if len(metadata["sharded"]) != parameter.ndim:
            raise ValueError(
                f"parameter {name!r} has ndim={parameter.ndim} but "
                f"{len(metadata['sharded'])} sharding entries"
            )
        if any(group is None for group in metadata["shared"] + metadata["reduce"]):
            raise ValueError(f"parameter {name!r} uses None outside sharded")

        groups: Iterable[str] = (
            group
            for values in metadata.values()
            for group in values
            if group is not None
        )
        unknown = set(groups) - known_groups
        if unknown:
            raise ValueError(
                f"parameter {name!r} references unknown groups {sorted(unknown)}"
            )

        for dimension, group in enumerate(metadata["sharded"]):
            if group is not None and parameter.shape[dimension] < 1:
                raise ValueError(
                    f"parameter {name!r} has an empty {group!r} shard on "
                    f"dimension {dimension}"
                )


def init_params_for_shared_weights(model: nn.Module) -> None:
    """Broadcast replicated parameter shards over their declared groups."""
    validate_param_comm_metadata(model)
    with torch.no_grad():
        for parameter in model.parameters():
            metadata = parameter.comm_metadata  # type: ignore[attr-defined]
            for comm_name in metadata["shared"]:
                if comm.get_size(comm_name) > 1:
                    dist.broadcast(
                        parameter,
                        src=comm.get_root(comm_name),
                        group=comm.get_group(comm_name),
                    )


def metadata_summary(model: nn.Module) -> list[str]:
    """Return a concise debug summary without printing from model code."""
    rows = ["parameter | shape | sharded | shared | reduce"]
    for name, parameter in model.named_parameters():
        metadata = getattr(parameter, "comm_metadata", None)
        rows.append(
            f"{name} | {tuple(parameter.shape)} | "
            f"{metadata['sharded']} | {metadata['shared']} | "
            f"{metadata['reduce']}"
        )
    return rows
