"""Low-level synchronous collective and split helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist

from . import comm


def compute_split_shapes(size: int, num_chunks: int) -> list[int]:
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    quotient, remainder = divmod(size, num_chunks)
    return [quotient + (index < remainder) for index in range(num_chunks)]


def compute_split_shapes_for_patching(
    size: int,
    num_chunks: int,
    patch_size: int,
) -> list[int]:
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    patch_count, remainder = divmod(size, patch_size)
    shapes = [
        patches * patch_size
        for patches in compute_split_shapes(patch_count, num_chunks)
    ]
    shapes[-1] += remainder
    return shapes


def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    num_chunks: int,
    sections: list[int] | None = None,
) -> list[torch.Tensor]:
    dim = dim % tensor.ndim
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if sections is None:
        sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    if len(sections) != num_chunks or sum(sections) != tensor.shape[dim]:
        raise ValueError(
            f"invalid sections={sections} for dimension size "
            f"{tensor.shape[dim]} and num_chunks={num_chunks}"
        )
    return list(torch.split(tensor, sections, dim=dim))


def _reduce(input_: torch.Tensor, comm_name: str) -> torch.Tensor:
    if comm.get_size(comm_name) == 1:
        return input_
    output = input_.contiguous().clone()
    dist.all_reduce(output, group=comm.get_group(comm_name))
    return output


def _split(
    input_: torch.Tensor,
    dim_: int,
    comm_name: str,
    sections: list[int] | None = None,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return input_
    chunks = split_tensor_along_dim(input_, dim_, size, sections)
    return chunks[comm.get_rank(comm_name)].contiguous()


def _gather(
    input_: torch.Tensor,
    dim_: int,
    shapes_: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return input_
    dim_ %= input_.ndim
    local = input_.contiguous()
    if shapes_ is None:
        shapes_ = [local.shape[dim_]] * size
    if len(shapes_) != size:
        raise ValueError(
            f"shapes length {len(shapes_)} does not match {comm_name} size {size}"
        )
    if shapes_[comm.get_rank(comm_name)] != local.shape[dim_]:
        raise ValueError(
            f"local {comm_name} shape {local.shape[dim_]} does not match "
            f"planned shape {shapes_[comm.get_rank(comm_name)]}"
        )

    max_shape = max(shapes_)
    if local.shape[dim_] < max_shape:
        pad_shape = list(local.shape)
        pad_shape[dim_] = max_shape - local.shape[dim_]
        local = torch.cat(
            (local, torch.zeros(pad_shape, dtype=local.dtype, device=local.device)),
            dim=dim_,
        )
    gathered = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(gathered, local, group=comm.get_group(comm_name))
    trimmed = [
        tensor.narrow(dim_, 0, shape)
        for tensor, shape in zip(gathered, shapes_, strict=True)
    ]
    return torch.cat(trimmed, dim=dim_).contiguous()


def _reduce_scatter(
    input_: torch.Tensor,
    dim_: int,
    shapes_: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    if comm.get_size(comm_name) == 1:
        return input_
    reduced = _reduce(input_, comm_name)
    return _split(reduced, dim_, comm_name, shapes_)
