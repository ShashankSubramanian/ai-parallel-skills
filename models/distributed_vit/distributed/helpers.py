"""Synchronous collective primitives indexed by communicator name."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist

from .. import comm


def compute_split_shapes(size: int, num_chunks: int) -> list[int]:
    if size < 0 or num_chunks < 1:
        raise ValueError("size must be non-negative and num_chunks positive")
    quotient, remainder = divmod(size, num_chunks)
    return [
        quotient + (1 if chunk < remainder else 0) for chunk in range(num_chunks)
    ]


def compute_split_shapes_for_patching(
    size: int, num_chunks: int, patch_size: int
) -> list[int]:
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    patch_units, remainder = divmod(size, patch_size)
    shapes = [
        units * patch_size
        for units in compute_split_shapes(patch_units, num_chunks)
    ]
    shapes[-1] += remainder
    return shapes


def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    num_chunks: int,
    sections: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    dim = dim % tensor.ndim
    if num_chunks < 1:
        raise ValueError("num_chunks must be positive")
    split_shapes = (
        list(sections)
        if sections is not None
        else compute_split_shapes(tensor.shape[dim], num_chunks)
    )
    if len(split_shapes) != num_chunks:
        raise ValueError("sections length must equal num_chunks")
    if sum(split_shapes) != tensor.shape[dim]:
        raise ValueError(
            f"sections sum to {sum(split_shapes)}, expected {tensor.shape[dim]}"
        )
    return list(torch.split(tensor, split_shapes, dim=dim))


def _reduce(input_: torch.Tensor, comm_name: str) -> torch.Tensor:
    if comm.get_size(comm_name) == 1:
        return input_
    output = input_.contiguous()
    dist.all_reduce(output, group=comm.get_group(comm_name))
    return output


def _split(
    input_: torch.Tensor,
    dim_: int,
    comm_name: str,
    sections: Sequence[int] | None = None,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return input_
    chunks = split_tensor_along_dim(input_, dim_, size, sections)
    return chunks[comm.get_rank(comm_name)].contiguous()


def _pad_along_dim(
    tensor: torch.Tensor, dim: int, target_size: int
) -> torch.Tensor:
    dim %= tensor.ndim
    if tensor.shape[dim] > target_size:
        raise ValueError("target size is smaller than tensor")
    if tensor.shape[dim] == target_size:
        return tensor.contiguous()
    shape = list(tensor.shape)
    shape[dim] = target_size - tensor.shape[dim]
    padding = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=dim).contiguous()


def _gather(
    input_: torch.Tensor,
    dim_: int,
    shapes_: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return input_
    dim_ %= input_.ndim
    shapes = (
        list(shapes_)
        if shapes_ is not None
        else [input_.shape[dim_]] * size
    )
    if len(shapes) != size:
        raise ValueError("gather shapes length must equal communicator size")
    local_rank = comm.get_rank(comm_name)
    if input_.shape[dim_] != shapes[local_rank]:
        raise ValueError(
            f"local gather size {input_.shape[dim_]} does not match "
            f"declared size {shapes[local_rank]}"
        )

    max_size = max(shapes)
    padded = _pad_along_dim(input_, dim_, max_size)
    gathered = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(gathered, padded, group=comm.get_group(comm_name))
    logical = [
        tensor.narrow(dim_, 0, shape) for tensor, shape in zip(gathered, shapes)
    ]
    return torch.cat(logical, dim=dim_).contiguous()


def _reduce_scatter(
    input_: torch.Tensor,
    dim_: int,
    shapes_: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    if comm.get_size(comm_name) == 1:
        return input_
    # This all-reduce + split formulation supports uneven shards on NCCL and
    # Gloo. It is semantically a reduce-scatter and can later be replaced by a
    # native equal-shape fast path without changing mappings.
    return _split(
        _reduce(input_.contiguous(), comm_name),
        dim_,
        comm_name,
        shapes_,
    )


def invert_permutation(sigma: Sequence[int]) -> list[int]:
    if sorted(sigma) != list(range(len(sigma))):
        raise ValueError("input is not a permutation")
    inverse = [0] * len(sigma)
    for index, value in enumerate(sigma):
        inverse[value] = index
    return inverse


def _distributed_roll(
    x: torch.Tensor,
    shift: int,
    shift_dim: int,
    comm_name: str,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return torch.roll(x, shift, shift_dim)

    local_size = x.shape[shift_dim]
    sizes: list[int | None] = [None] * size
    dist.all_gather_object(
        sizes, local_size, group=comm.get_group(comm_name)
    )
    split_sizes = [int(value) for value in sizes]
    logical = _gather(x, shift_dim, split_sizes, comm_name)
    rolled = torch.roll(logical, shift, shift_dim)
    return _split(rolled, shift_dim, comm_name, split_sizes)


def _transpose(
    tensor: torch.Tensor,
    dim0: int,
    dim1: int,
    dim0_split_sizes: Sequence[int],
    dim1_split_sizes: Sequence[int],
    comm_name: str,
    async_op: bool = False,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    if size == 1:
        return tensor
    if async_op:
        raise NotImplementedError("asynchronous distributed transpose is unsupported")
    if len(dim0_split_sizes) != size or len(dim1_split_sizes) != size:
        raise ValueError("transpose split lists must match communicator size")

    dim0 %= tensor.ndim
    dim1 %= tensor.ndim
    local_rank = comm.get_rank(comm_name)
    if tensor.shape[dim0] != sum(dim0_split_sizes):
        raise ValueError("dim0 split sizes do not match local tensor")
    if tensor.shape[dim1] != dim1_split_sizes[local_rank]:
        raise ValueError("local dim1 does not match declared split size")

    send = split_tensor_along_dim(tensor, dim0, size, dim0_split_sizes)
    receive: list[torch.Tensor] = []
    for source_rank in range(size):
        shape = list(tensor.shape)
        shape[dim0] = dim0_split_sizes[local_rank]
        shape[dim1] = dim1_split_sizes[source_rank]
        receive.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
    dist.all_to_all(
        receive,
        [chunk.contiguous() for chunk in send],
        group=comm.get_group(comm_name),
    )
    return torch.cat(receive, dim=dim1).contiguous()
