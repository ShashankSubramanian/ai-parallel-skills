"""AdamW wrapper with explicit FP32 master parameters and optimizer state."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class FP32MasterAdamW:
    """Update BF16 model shards through FP32 master copies."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.model_parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        self.master_parameters = [
            nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for parameter in self.model_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.master_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    @torch.no_grad()
    def step(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters
        ):
            if model_parameter.grad is None:
                master_parameter.grad = None
            else:
                master_parameter.grad = model_parameter.grad.detach().float()
        self.optimizer.step()
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters
        ):
            model_parameter.copy_(
                master_parameter.to(dtype=model_parameter.dtype)
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.model_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)
