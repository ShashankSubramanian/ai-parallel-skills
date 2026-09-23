"""Autograd-aware collective mappings and DDP reduction integration."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import comm
from .helpers import _gather, _reduce, _reduce_scatter, _split


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_: torch.Tensor, comm_name: str) -> torch.Tensor:
        return input_

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.comm_name = inputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return _reduce(grad_output, ctx.comm_name), None


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_: torch.Tensor, comm_name: str) -> torch.Tensor:
        return _reduce(input_, comm_name)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.comm_name = inputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _AllGatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        input_: torch.Tensor,
        dim: int,
        shapes: list[int] | None,
        comm_name: str,
    ) -> torch.Tensor:
        return _gather(input_, dim, shapes, comm_name)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return _split(grad_output, ctx.dim, ctx.comm_name, ctx.shapes), None, None, None


class _SplitForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        input_: torch.Tensor,
        dim: int,
        shapes: list[int] | None,
        comm_name: str,
    ) -> torch.Tensor:
        return _split(input_, dim, comm_name, shapes)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name), None, None, None


class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        input_: torch.Tensor,
        dim: int,
        shapes: list[int] | None,
        comm_name: str,
    ) -> torch.Tensor:
        return _reduce_scatter(input_, dim, shapes, comm_name)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name), None, None, None


class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        input_: torch.Tensor,
        dim: int,
        shapes: list[int] | None,
        comm_name: str,
    ) -> torch.Tensor:
        return _gather(input_, dim, shapes, comm_name)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return (
            _reduce_scatter(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


def identity_forward_allreduce_backward(
    input_: torch.Tensor, comm_name: str
) -> torch.Tensor:
    return _IdentityForwardAllReduceBackward.apply(input_, comm_name)


def allreduce_forward_identity_backward(
    input_: torch.Tensor, comm_name: str
) -> torch.Tensor:
    return _AllReduceForwardIdentityBackward.apply(input_, comm_name)


def allgather_forward_split_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _AllGatherForwardSplitBackward.apply(input_, dim, shapes, comm_name)


def split_forward_allgather_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _SplitForwardAllGatherBackward.apply(input_, dim, shapes, comm_name)


def reducescatter_forward_allgather_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _ReduceScatterForwardAllGatherBackward.apply(
        input_, dim, shapes, comm_name
    )


def allgather_forward_reducescatter_backward(
    input_: torch.Tensor,
    dim: int,
    shapes: list[int] | None,
    comm_name: str,
) -> torch.Tensor:
    return _AllGatherForwardReduceScatterBackward.apply(
        input_, dim, shapes, comm_name
    )


@dataclass(frozen=True)
class _DDPHookState:
    reduction_order: tuple[str, ...]


def _ddp_reduction_hook(
    state: _DDPHookState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    buffer = bucket.buffer()
    if comm.get_size("dp") > 1:
        dist.all_reduce(buffer, group=comm.get_group("dp"))
        buffer.div_(comm.get_size("dp"))

    parameters = bucket.parameters()
    gradients = bucket.gradients()
    for group_name in state.reduction_order:
        if comm.get_size(group_name) == 1:
            continue
        selected = [
            gradient
            for parameter, gradient in zip(parameters, gradients, strict=True)
            if group_name in parameter.comm_metadata["reduce"]
        ]
        if not selected:
            continue
        flattened = torch.cat([gradient.reshape(-1) for gradient in selected])
        dist.all_reduce(flattened, group=comm.get_group(group_name))
        offset = 0
        for gradient in selected:
            count = gradient.numel()
            gradient.copy_(flattened.narrow(0, offset, count).view_as(gradient))
            offset += count

    future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    future.set_result(buffer)
    return future


def init_ddp_model_and_reduction_hooks(
    model: torch.nn.Module,
    *,
    reduction_order: tuple[str, ...] = ("cp",),
) -> torch.nn.Module:
    """Wrap with DDP and consume all effective non-DP reduction metadata."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in trainable:
        if not hasattr(parameter, "comm_metadata"):
            raise ValueError("all trainable parameters must have comm_metadata")

    effective = {
        tuple(
            group_name
            for group_name in reduction_order
            if group_name in parameter.comm_metadata["reduce"]
            and comm.get_size(group_name) > 1
        )
        for parameter in trainable
    }
    if comm.get_world_size() == 1:
        if any(effective):
            raise RuntimeError("nontrivial gradient reductions require distributed setup")
        return model

    device_ids = None
    output_device = None
    if next(model.parameters()).device.type == "cuda":
        device_ids = [comm.get_local_rank()]
        output_device = comm.get_local_rank()
    ddp_model = DistributedDataParallel(
        model,
        process_group=comm.get_group("dp"),
        device_ids=device_ids,
        output_device=output_device,
        forward_sync_buffers=False,
        find_unused_parameters=False,
    )
    if effective != {()}:
        ddp_model.register_comm_hook(
            _DDPHookState(reduction_order=reduction_order),
            _ddp_reduction_hook,
        )
    return ddp_model
