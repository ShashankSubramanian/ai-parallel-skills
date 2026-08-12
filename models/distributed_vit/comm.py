"""Process-group registry for the distributed ViT."""

from __future__ import annotations

import os
from math import prod
from typing import Iterable, Sequence

import torch
import torch.distributed as dist

from .rank_generator import RankGenerator

_COMM_GROUPS: dict[str, dist.ProcessGroup] = {}
_COMM_RANKS: dict[str, list[int]] = {}
_OWNS_DEFAULT_GROUP = False


def initialize(
    *,
    tp_size: int = 1,
    cp_size: int = 1,
    order: str = "tp-cp",
    group_tokens: Sequence[str] = ("tp", "cp"),
    backend: str | None = None,
    init_method: str | None = None,
) -> None:
    """Initialize the default process group and planned orthogonal groups."""
    global _OWNS_DEFAULT_GROUP

    if _COMM_GROUPS or _COMM_RANKS:
        raise RuntimeError("communication groups are already initialized")

    dimensions = {"tp": tp_size, "cp": cp_size}
    expected_world_size = prod(dimensions.values())
    if expected_world_size < 1:
        raise ValueError("mesh sizes must be positive")

    if not dist.is_available():
        if expected_world_size != 1:
            raise RuntimeError("torch.distributed is unavailable for a multi-rank mesh")
        _initialize_single_rank(group_tokens)
        return

    if not dist.is_initialized():
        if expected_world_size == 1 and "RANK" not in os.environ:
            _initialize_single_rank(group_tokens)
            return
        selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        kwargs: dict[str, str] = {"backend": selected_backend}
        if init_method is not None:
            kwargs["init_method"] = init_method
        dist.init_process_group(**kwargs)
        _OWNS_DEFAULT_GROUP = True

    world_size = dist.get_world_size()
    if world_size != expected_world_size:
        raise ValueError(
            f"world_size={world_size} does not match tp*cp={expected_world_size}"
        )

    rank = dist.get_rank()
    generator = RankGenerator(dimensions=dimensions, order=order)
    for token in group_tokens:
        found_current_group = False
        for ranks in generator.get_ranks(token):
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                _COMM_GROUPS[token] = group
                _COMM_RANKS[token] = ranks
                found_current_group = True
        if not found_current_group:
            raise RuntimeError(f"rank {rank} was not assigned to group {token!r}")


def _initialize_single_rank(group_tokens: Sequence[str]) -> None:
    for token in group_tokens:
        _COMM_RANKS[token] = [0]


def destroy() -> None:
    """Clear project groups and destroy a project-owned default group."""
    global _OWNS_DEFAULT_GROUP
    _COMM_GROUPS.clear()
    _COMM_RANKS.clear()
    if _OWNS_DEFAULT_GROUP and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _OWNS_DEFAULT_GROUP = False


def get_names() -> Iterable[str]:
    return tuple(_COMM_RANKS)


def is_initialized(comm_name: str) -> bool:
    return comm_name in _COMM_RANKS


def get_group(comm_name: str) -> dist.ProcessGroup:
    if comm_name not in _COMM_RANKS:
        raise RuntimeError(f"communication group {comm_name!r} was not built")
    if get_size(comm_name) == 1:
        # Collectives must bypass this synthetic single-rank group.
        raise RuntimeError(
            f"communication group {comm_name!r} is a single-rank no-op"
        )
    return _COMM_GROUPS[comm_name]


def get_size(comm_name: str) -> int:
    return len(_COMM_RANKS.get(comm_name, [0]))


def get_rank(comm_name: str) -> int:
    ranks = _COMM_RANKS.get(comm_name)
    if not ranks:
        return 0
    world_rank = get_world_rank()
    try:
        return ranks.index(world_rank)
    except ValueError as error:
        raise RuntimeError(
            f"world rank {world_rank} is not in communication group {comm_name!r}"
        ) from error


def get_root(comm_name: str) -> int:
    return min(_COMM_RANKS.get(comm_name, [0]))


def get_ranks(comm_name: str) -> list[int]:
    return list(_COMM_RANKS.get(comm_name, [0]))


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
