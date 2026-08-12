"""Differentiable collective mappings used at model ownership boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.hooks import RemovableHandle

from .. import comm
from .helpers import (
    _distributed_roll,
    _gather,
    _reduce,
    _reduce_scatter,
    _split,
    _transpose,
)


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm_name: str) -> torch.Tensor:
        ctx.comm_name = comm_name
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _reduce(grad_output, ctx.comm_name), None


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm_name: str) -> torch.Tensor:
        return _reduce(input_, comm_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _BroadcastForwardReduceToRootBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, comm_name: str) -> torch.Tensor:
        ctx.comm_name = comm_name
        ctx.root = comm.get_root(comm_name)
        if comm.get_size(comm_name) == 1:
            return input_
        output = input_.contiguous().clone()
        dist.broadcast(
            output,
            src=ctx.root,
            group=comm.get_group(comm_name),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if comm.get_size(ctx.comm_name) == 1:
            return grad_output, None
        reduced = _reduce(grad_output, ctx.comm_name)
        if comm.get_world_rank() != ctx.root:
            reduced = torch.zeros_like(reduced)
        return reduced, None


class _AllGatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        dim: int,
        shapes: tuple[int, ...] | None,
        comm_name: str,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.shapes = shapes
        ctx.comm_name = comm_name
        return _gather(input_, dim, shapes, comm_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _split(grad_output, ctx.dim, ctx.comm_name, ctx.shapes),
            None,
            None,
            None,
        )


class _SplitForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        dim: int,
        shapes: tuple[int, ...] | None,
        comm_name: str,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.shapes = shapes
        ctx.comm_name = comm_name
        return _split(input_, dim, comm_name, shapes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        dim: int,
        shapes: tuple[int, ...] | None,
        comm_name: str,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.shapes = shapes
        ctx.comm_name = comm_name
        return _reduce_scatter(input_, dim, shapes, comm_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        dim: int,
        shapes: tuple[int, ...] | None,
        comm_name: str,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.shapes = shapes
        ctx.comm_name = comm_name
        return _gather(input_, dim, shapes, comm_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _reduce_scatter(
                grad_output, ctx.dim, ctx.shapes, ctx.comm_name
            ),
            None,
            None,
            None,
        )


class RollForwardReverseRollBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        shift: int,
        dim: int,
        comm_name: str,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.dim = dim
        ctx.comm_name = comm_name
        return _distributed_roll(x, shift, dim, comm_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _distributed_roll(
                grad_output, -ctx.shift, ctx.dim, ctx.comm_name
            ),
            None,
            None,
            None,
        )


class TransposeForwardTransposeBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dim0: int,
        dim1: int,
        dim0_split_sizes: tuple[int, ...],
        dim1_split_sizes: tuple[int, ...],
        comm_name: str,
    ) -> torch.Tensor:
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        ctx.dim0_split_sizes = dim0_split_sizes
        ctx.dim1_split_sizes = dim1_split_sizes
        ctx.comm_name = comm_name
        return _transpose(
            x,
            dim0,
            dim1,
            dim0_split_sizes,
            dim1_split_sizes,
            comm_name,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _transpose(
                grad_output,
                ctx.dim1,
                ctx.dim0,
                ctx.dim1_split_sizes,
                ctx.dim0_split_sizes,
                ctx.comm_name,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def _shape_tuple(shapes: Sequence[int] | None) -> tuple[int, ...] | None:
    return tuple(shapes) if shapes is not None else None


def identity_forward_allreduce_backward(
    input_: torch.Tensor, comm_name: str
) -> torch.Tensor:
    return _IdentityForwardAllReduceBackward.apply(input_, comm_name)


def allreduce_forward_identity_backward(
    input_: torch.Tensor, comm_name: str
) -> torch.Tensor:
    return _AllReduceForwardIdentityBackward.apply(input_, comm_name)


def broadcast_forward_reduce_to_root_backward(
    input_: torch.Tensor, comm_name: str
) -> torch.Tensor:
    return _BroadcastForwardReduceToRootBackward.apply(input_, comm_name)


def allgather_forward_split_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _AllGatherForwardSplitBackward.apply(
        input_, dim, _shape_tuple(shapes), comm_name
    )


def split_forward_allgather_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _SplitForwardAllGatherBackward.apply(
        input_, dim, _shape_tuple(shapes), comm_name
    )


def reducescatter_forward_allgather_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _ReduceScatterForwardAllGatherBackward.apply(
        input_, dim, _shape_tuple(shapes), comm_name
    )


def allgather_forward_reducescatter_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: Sequence[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _AllGatherForwardReduceScatterBackward.apply(
        input_, dim, _shape_tuple(shapes), comm_name
    )


def roll_forward_reverseroll_backward(
    x: torch.Tensor,
    shifts: Sequence[int],
    dims: Sequence[int],
    comm_names: Sequence[str],
) -> torch.Tensor:
    if not (len(shifts) == len(dims) == len(comm_names)):
        raise ValueError("roll shifts, dims, and comm_names must have equal lengths")
    output = x
    for shift, dim, comm_name in zip(shifts, dims, comm_names):
        output = RollForwardReverseRollBackward.apply(
            output, shift, dim, comm_name
        )
    return output


def transpose_forward_transpose_backward(
    x: torch.Tensor,
    dims: tuple[int, int],
    dim0_split_sizes: Sequence[int],
    dim1_split_sizes: Sequence[int],
    comm_name: str,
) -> torch.Tensor:
    return TransposeForwardTransposeBackward.apply(
        x,
        dims[0],
        dims[1],
        tuple(dim0_split_sizes),
        tuple(dim1_split_sizes),
        comm_name,
    )


def register_model_parallel_gradient_hooks(
    model: nn.Module,
) -> list[RemovableHandle]:
    """All-reduce each declared non-DP parameter gradient exactly once."""
    handles: list[RemovableHandle] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        metadata = parameter.comm_metadata  # type: ignore[attr-defined]
        reduce_groups = tuple(dict.fromkeys(metadata["reduce"]))

        if not reduce_groups:
            continue

        def reduce_gradient(
            gradient: torch.Tensor,
            groups: tuple[str, ...] = reduce_groups,
        ) -> torch.Tensor:
            reduced = gradient
            for comm_name in groups:
                reduced = _reduce(reduced, comm_name)
            return reduced

        handles.append(parameter.register_hook(reduce_gradient))
    return handles
