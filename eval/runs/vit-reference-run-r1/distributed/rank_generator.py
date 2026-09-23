"""Pure Megatron-style orthogonal rank-group generation."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod


def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: list[int],
    mask: list[bool],
) -> list[list[int]]:
    """Generate groups where masked dimensions vary and others stay fixed.

    ``parallel_size`` is ordered from fastest- to slowest-varying dimension.
    """
    if len(parallel_size) != len(mask):
        raise ValueError("parallel_size and mask must have equal length")
    if any(size < 1 for size in parallel_size):
        raise ValueError(f"parallel sizes must be positive, got {parallel_size}")
    if prod(parallel_size) != world_size:
        raise ValueError(
            f"world_size={world_size} does not match mesh product "
            f"{prod(parallel_size)}"
        )

    strides: list[int] = []
    stride = 1
    for size in parallel_size:
        strides.append(stride)
        stride *= size

    groups: dict[tuple[int, ...], list[int]] = {}
    for global_rank in range(world_size):
        coordinates = tuple(
            (global_rank // dim_stride) % dim_size
            for dim_stride, dim_size in zip(strides, parallel_size, strict=True)
        )
        fixed_coordinates = tuple(
            coordinate
            for coordinate, is_varying in zip(coordinates, mask, strict=True)
            if not is_varying
        )
        groups.setdefault(fixed_coordinates, []).append(global_rank)
    return list(groups.values())


class RankGenerator:
    """Generate orthogonal groups for a named hyperrectangle mesh."""

    def __init__(self, dimensions: Mapping[str, int], order: str) -> None:
        if not dimensions:
            raise ValueError("dimensions must not be empty")
        self.dimensions = dict(dimensions)
        if any(size < 1 for size in self.dimensions.values()):
            raise ValueError(f"mesh dimensions must be positive: {self.dimensions}")

        self.order = tuple(part for part in order.split("-") if part)
        if len(set(self.order)) != len(self.order):
            raise ValueError(f"rank order contains duplicates: {order!r}")
        unknown = set(self.order) - set(self.dimensions)
        if unknown:
            raise ValueError(f"rank order names unknown dimensions: {sorted(unknown)}")
        missing_active = {
            name
            for name, size in self.dimensions.items()
            if size > 1 and name not in self.order
        }
        if missing_active:
            raise ValueError(
                f"rank order omits active dimensions: {sorted(missing_active)}"
            )
        missing = set(self.dimensions) - set(self.order)
        if missing:
            raise ValueError(
                "rank order must include size-one dimensions for deterministic "
                f"topology, missing {sorted(missing)}"
            )

        self.world_size = prod(self.dimensions.values())
        self._ordered_sizes = [self.dimensions[name] for name in self.order]

    def get_ranks(self, token: str) -> list[list[int]]:
        token_names = tuple(part for part in token.split("-") if part)
        if not token_names:
            raise ValueError("group token must not be empty")
        if len(set(token_names)) != len(token_names):
            raise ValueError(f"group token contains duplicates: {token!r}")
        unknown = set(token_names) - set(self.dimensions)
        if unknown:
            raise ValueError(f"group token names unknown dimensions: {sorted(unknown)}")
        mask = [name in token_names for name in self.order]
        return generate_masked_orthogonal_rank_groups(
            self.world_size,
            self._ordered_sizes,
            mask,
        )
