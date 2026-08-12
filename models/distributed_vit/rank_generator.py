"""Pure rank-math utilities for orthogonal process groups."""

from __future__ import annotations

from collections import defaultdict
from math import prod
from typing import Mapping, Sequence


def _coordinates(rank: int, sizes: Sequence[int]) -> tuple[int, ...]:
    coordinates: list[int] = []
    remainder = rank
    for size in sizes:
        coordinates.append(remainder % size)
        remainder //= size
    return tuple(coordinates)


def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: Sequence[int],
    mask: Sequence[bool],
) -> list[list[int]]:
    """Generate hyperrectangle groups whose masked coordinates vary."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if len(parallel_size) != len(mask):
        raise ValueError("parallel_size and mask must have equal lengths")
    if any(size < 1 for size in parallel_size):
        raise ValueError("parallel dimensions must be positive")
    if prod(parallel_size) != world_size:
        raise ValueError(
            f"world_size={world_size} does not match mesh product "
            f"{prod(parallel_size)}"
        )

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for rank in range(world_size):
        coords = _coordinates(rank, parallel_size)
        fixed_coords = tuple(
            coordinate for coordinate, varies in zip(coords, mask) if not varies
        )
        groups[fixed_coords].append(rank)
    return list(groups.values())


class RankGenerator:
    """Generate named orthogonal rank groups for a device mesh."""

    def __init__(self, dimensions: Mapping[str, int], order: str) -> None:
        if not dimensions:
            raise ValueError("dimensions must not be empty")
        if any(size < 1 for size in dimensions.values()):
            raise ValueError("all dimension sizes must be positive")

        names = order.split("-") if order else []
        if len(names) != len(set(names)):
            raise ValueError(f"rank order contains duplicate dimensions: {order}")
        unknown = set(names) - set(dimensions)
        missing = set(dimensions) - set(names)
        if unknown or missing:
            raise ValueError(
                f"rank order must contain each dimension exactly once; "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )

        self.dimensions = dict(dimensions)
        self.order = order
        self._names = names
        self._sizes = [self.dimensions[name] for name in names]
        self.world_size = prod(self._sizes)

    def get_ranks(self, token: str) -> list[list[int]]:
        """Return groups in which dimensions named by ``token`` vary."""
        token_names = token.split("-") if token else []
        if not token_names or len(token_names) != len(set(token_names)):
            raise ValueError(f"invalid group token: {token!r}")
        unknown = set(token_names) - set(self._names)
        if unknown:
            raise ValueError(
                f"group token {token!r} contains unknown dimensions "
                f"{sorted(unknown)}"
            )
        mask = [name in token_names for name in self._names]
        return generate_masked_orthogonal_rank_groups(
            self.world_size, self._sizes, mask
        )
