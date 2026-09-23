"""Distributed initialization and process-group registry."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from math import prod
from typing import Any

import torch
import torch.distributed as dist

from .rank_generator import RankGenerator

DEFAULT_DIMENSIONS = {"dp": 1, "tp": 4, "cp": 8}
DEFAULT_ORDER = "tp-cp-dp"
DEFAULT_GROUP_TOKENS = ("dp", "tp", "cp", "dp-tp", "dp-cp")

_COMM_GROUPS: dict[str, Any | None] = {}
_COMM_RANKS: dict[str, list[int]] = {}
_CREATED_GROUPS: list[Any] = []
_DIMENSIONS: dict[str, int] = {}
_ORDER = ""
_OWNS_DEFAULT_GROUP = False


def initialize(
    *,
    dimensions: Mapping[str, int] | None = None,
    order: str = DEFAULT_ORDER,
    group_tokens: Sequence[str] = DEFAULT_GROUP_TOKENS,
    backend: str | None = None,
    debug: bool = False,
) -> None:
    """Initialize the default group and all groups in the approved topology."""
    global _DIMENSIONS, _ORDER, _OWNS_DEFAULT_GROUP
    if _COMM_GROUPS:
        raise RuntimeError("communication groups are already initialized")

    dimensions = dict(dimensions or DEFAULT_DIMENSIONS)
    expected_world_size = prod(dimensions.values())
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size != expected_world_size:
        raise ValueError(
            f"WORLD_SIZE={launched_world_size} does not match mesh product "
            f"{expected_world_size} for dimensions={dimensions}"
        )

    rank_generator = RankGenerator(dimensions, order)
    tokens = tuple(dict.fromkeys(group_tokens))
    for token in tokens:
        rank_generator.get_ranks(token)

    _DIMENSIONS = dimensions
    _ORDER = order

    if expected_world_size == 1 and not dist.is_initialized():
        for token in tokens:
            _COMM_GROUPS[token] = None
            _COMM_RANKS[token] = [0]
        if debug:
            print(topology_summary(), flush=True)
        return

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is unavailable")
        torch.cuda.set_device(get_local_rank())

    if dist.is_initialized():
        if dist.get_world_size() != expected_world_size:
            raise ValueError(
                f"initialized world size {dist.get_world_size()} does not match "
                f"mesh product {expected_world_size}"
            )
        _OWNS_DEFAULT_GROUP = False
    else:
        dist.init_process_group(backend=backend, init_method="env://")
        _OWNS_DEFAULT_GROUP = True

    world_rank = dist.get_rank()
    for token in tokens:
        found_current = False
        for ranks in rank_generator.get_ranks(token):
            group = dist.new_group(ranks=ranks, backend=backend)
            if world_rank in ranks:
                _CREATED_GROUPS.append(group)
                _COMM_GROUPS[token] = group
                _COMM_RANKS[token] = ranks
                found_current = True
        if not found_current:
            raise RuntimeError(f"rank {world_rank} was not assigned to group {token!r}")

    if debug and world_rank == 0:
        print(topology_summary(), flush=True)


def destroy() -> None:
    """Destroy groups created by this module and clear all topology state."""
    global _ORDER, _OWNS_DEFAULT_GROUP
    if dist.is_available() and dist.is_initialized():
        for group in reversed(_CREATED_GROUPS):
            try:
                dist.destroy_process_group(group)
            except ValueError:
                pass
        if _OWNS_DEFAULT_GROUP:
            dist.destroy_process_group()
    _CREATED_GROUPS.clear()
    _COMM_GROUPS.clear()
    _COMM_RANKS.clear()
    _DIMENSIONS.clear()
    _ORDER = ""
    _OWNS_DEFAULT_GROUP = False


def get_names() -> Iterable[str]:
    return tuple(_COMM_GROUPS)


def is_initialized(comm_name: str) -> bool:
    return comm_name in _COMM_GROUPS


def _require_name(comm_name: str) -> None:
    if comm_name not in _COMM_GROUPS:
        raise KeyError(
            f"communication group {comm_name!r} is not registered; "
            f"available={sorted(_COMM_GROUPS)}"
        )


def get_group(comm_name: str) -> Any | None:
    _require_name(comm_name)
    return _COMM_GROUPS[comm_name]


def get_size(comm_name: str) -> int:
    _require_name(comm_name)
    return len(_COMM_RANKS[comm_name])


def get_rank(comm_name: str) -> int:
    _require_name(comm_name)
    return _COMM_RANKS[comm_name].index(get_world_rank())


def get_root(comm_name: str) -> int:
    _require_name(comm_name)
    return min(_COMM_RANKS[comm_name])


def get_ranks(comm_name: str) -> list[int]:
    _require_name(comm_name)
    return list(_COMM_RANKS[comm_name])


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_world_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return get_world_rank() % torch.cuda.device_count()
    return 0


def get_dimensions() -> dict[str, int]:
    return dict(_DIMENSIONS)


def get_order() -> str:
    return _ORDER


def topology_summary() -> str:
    groups = ", ".join(
        f"{name}={_COMM_RANKS[name]}" for name in sorted(_COMM_RANKS)
    )
    return (
        f"mesh={_DIMENSIONS} order={_ORDER!r} world_rank={get_world_rank()} "
        f"groups: {groups}"
    )
