"""Parameter communication metadata validation and shared initialization."""

from __future__ import annotations

import torch
import torch.distributed as dist

from . import comm


def set_comm_metadata(
    parameter: torch.nn.Parameter,
    *,
    sharded: list[str | None],
    shared: list[str],
    reduce: list[str],
) -> None:
    parameter.comm_metadata = {
        "sharded": list(sharded),
        "shared": list(shared),
        "reduce": list(reduce),
    }


def validate_param_comm_metadata(
    model: torch.nn.Module,
    *,
    print_summary: bool = False,
) -> None:
    valid_groups = set(comm.get_names())
    required_keys = {"sharded", "shared", "reduce"}
    summary: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not hasattr(parameter, "comm_metadata"):
            raise ValueError(f"{name}: missing comm_metadata")
        metadata = parameter.comm_metadata
        if set(metadata) != required_keys:
            raise ValueError(
                f"{name}: metadata keys must be {sorted(required_keys)}, "
                f"got {sorted(metadata)}"
            )
        if not all(isinstance(metadata[key], list) for key in required_keys):
            raise TypeError(f"{name}: every metadata field must be a list")
        if len(metadata["sharded"]) != parameter.ndim:
            raise ValueError(
                f"{name}: sharded metadata length {len(metadata['sharded'])} "
                f"does not match parameter rank {parameter.ndim}"
            )
        for group_name in metadata["sharded"]:
            if group_name is not None and group_name not in valid_groups:
                raise ValueError(f"{name}: unknown sharded group {group_name!r}")
        for field in ("shared", "reduce"):
            if any(group_name is None for group_name in metadata[field]):
                raise ValueError(f"{name}: None is invalid in metadata[{field!r}]")
            unknown = set(metadata[field]) - valid_groups
            if unknown:
                raise ValueError(
                    f"{name}: unknown {field} groups {sorted(unknown)}"
                )
        summary.append(
            f"{name} | {tuple(parameter.shape)} | {metadata['sharded']} | "
            f"{metadata['shared']} | {metadata['reduce']}"
        )
    if print_summary and comm.get_world_rank() == 0:
        print("parameter | shape | sharded | shared | reduce", flush=True)
        print("\n".join(summary), flush=True)


@torch.no_grad()
def init_params_for_shared_weights(model: torch.nn.Module) -> None:
    validate_param_comm_metadata(model)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        for group_name in parameter.comm_metadata["shared"]:
            if comm.get_size(group_name) > 1:
                dist.broadcast(
                    parameter,
                    src=comm.get_root(group_name),
                    group=comm.get_group(group_name),
                )
