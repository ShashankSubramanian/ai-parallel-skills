"""Baseline/distributed forward, input-gradient, and wgrad equivalence.

Launch examples:
  python -m unittest models.distributed_vit.tests.test_equivalence
  TP_SIZE=2 CP_SIZE=2 torchrun --standalone --nproc-per-node=4 \
    -m unittest models.distributed_vit.tests.test_equivalence

Unlike the generic skeleton's externally split local input, this model accepts
the full logical image and owns a differentiable CP split at its patch boundary.
The full input is therefore the leaf on every rank, and its backward mapping
reconstructs the full logical input gradient directly.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from torch import nn

from models.vit import VisionTransformer as BaselineVisionTransformer
from models.distributed_vit import comm
from models.distributed_vit.distributed.helpers import _gather
from models.distributed_vit.distributed.mappings import (
    register_model_parallel_gradient_hooks,
)
from models.distributed_vit.metadata import (
    init_params_for_shared_weights,
    validate_param_comm_metadata,
)
from models.distributed_vit.model import VisionTransformer

SEED = 333
RTOL = 1e-4
ATOL = 1e-5

CONFIG = {
    "height": 8,
    "width": 12,
    "channels": 3,
    "patch_size": 2,
    "dim": 16,
    "depth": 2,
    "heads": 4,
    "mlp_ratio": 2.0,
    "num_classes": 7,
}


def diagnostic(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rank: int,
) -> str:
    if expected.shape != actual.shape:
        return (
            f"{name}: rank={rank} expected_shape={tuple(expected.shape)} "
            f"actual_shape={tuple(actual.shape)}"
        )
    difference = actual.detach().float() - expected.detach().float()
    denominator = expected.detach().float().norm().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (
        f"{name}: rank={rank} shape={tuple(actual.shape)} "
        f"max_abs={difference.abs().max().item():.6e} "
        f"rel_l2={(difference.norm() / denominator).item():.6e} "
        f"mesh=tp{comm.get_size('tp')}xcp{comm.get_size('cp')} "
        f"rtol={RTOL} atol={ATOL}"
    )


def assert_tensor_close(
    test: unittest.TestCase,
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    rank: int,
) -> None:
    message = diagnostic(name, expected, actual, rank=rank)
    test.assertEqual(tuple(actual.shape), tuple(expected.shape), message)
    test.assertTrue(torch.isfinite(expected).all().item(), message)
    test.assertTrue(torch.isfinite(actual).all().item(), message)
    try:
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError as error:
        raise AssertionError(message) from error


def build_baseline(device: torch.device) -> BaselineVisionTransformer:
    # The immutable baseline constructor is square-only, but its forward accepts
    # rectangular inputs once the learned positional table has the right length.
    model = BaselineVisionTransformer(
        image_size=CONFIG["height"],
        patch_size=CONFIG["patch_size"],
        in_channels=CONFIG["channels"],
        num_classes=CONFIG["num_classes"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        heads=CONFIG["heads"],
        mlp_ratio=CONFIG["mlp_ratio"],
        dropout=0.0,
    )
    tokens = (
        (CONFIG["height"] // CONFIG["patch_size"])
        * (CONFIG["width"] // CONFIG["patch_size"])
        + 1
    )
    model.pos_embed = nn.Parameter(torch.empty(1, tokens, CONFIG["dim"]))
    nn.init.trunc_normal_(model.pos_embed, std=0.02)
    return model.to(device)


def build_distributed(device: torch.device) -> VisionTransformer:
    return VisionTransformer(
        image_size=(CONFIG["height"], CONFIG["width"]),
        patch_size=CONFIG["patch_size"],
        in_channels=CONFIG["channels"],
        num_classes=CONFIG["num_classes"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        heads=CONFIG["heads"],
        mlp_ratio=CONFIG["mlp_ratio"],
        dropout=0.0,
    ).to(device)


def _slice(
    tensor: torch.Tensor,
    dim: int,
    comm_name: str,
    shapes: list[int] | None = None,
) -> torch.Tensor:
    size = comm.get_size(comm_name)
    rank = comm.get_rank(comm_name)
    if shapes is None:
        if tensor.shape[dim] % size:
            raise AssertionError(
                f"cannot evenly shard shape={tuple(tensor.shape)} dim={dim} "
                f"over {comm_name} size={size}"
            )
        shapes = [tensor.shape[dim] // size] * size
    start = sum(shapes[:rank])
    return tensor.narrow(dim, start, shapes[rank])


def _is_qkv(name: str) -> bool:
    return name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias")


@torch.no_grad()
def copy_logical_parameters(
    baseline: nn.Module, distributed_model: VisionTransformer
) -> None:
    baseline_parameters = dict(baseline.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    if baseline_parameters.keys() != distributed_parameters.keys():
        raise AssertionError(
            "parameter-name mismatch: "
            f"missing={baseline_parameters.keys() - distributed_parameters.keys()}, "
            f"extra={distributed_parameters.keys() - baseline_parameters.keys()}"
        )

    for name, local_parameter in distributed_parameters.items():
        logical = baseline_parameters[name].detach()
        if _is_qkv(name):
            # Convert baseline [3e, e] / [3e] storage to the distributed
            # [3, heads/tp, head_dim, ...] storage before slicing heads.
            head_dim = CONFIG["dim"] // CONFIG["heads"]
            logical = logical.reshape(
                3,
                CONFIG["heads"],
                head_dim,
                *logical.shape[1:],
            )
            if comm.get_size("tp") > 1:
                logical = _slice(logical, 1, "tp")
            logical = logical.reshape(local_parameter.shape)
        else:
            for dim, comm_name in enumerate(
                local_parameter.comm_metadata["sharded"]
            ):
                if comm_name is None:
                    continue
                shapes = (
                    distributed_model.sequence_shapes
                    if name == "pos_embed" and dim == 1
                    else None
                )
                logical = _slice(logical, dim, comm_name, shapes)
        if logical.shape != local_parameter.shape:
            raise AssertionError(
                f"{name}: logical shard={tuple(logical.shape)} "
                f"local={tuple(local_parameter.shape)} "
                f"metadata={local_parameter.comm_metadata}"
            )
        local_parameter.copy_(logical)


def reconstruct_gradient(
    name: str,
    parameter: nn.Parameter,
    distributed_model: VisionTransformer,
) -> torch.Tensor:
    if parameter.grad is None:
        raise AssertionError(f"{name}: missing local parameter gradient")
    gradient = parameter.grad.detach()
    if _is_qkv(name):
        if comm.get_size("tp") > 1:
            gradient = _gather(gradient, 1, None, "tp")
        return gradient.reshape(3 * CONFIG["dim"], *gradient.shape[3:])

    for dim, comm_name in enumerate(parameter.comm_metadata["sharded"]):
        if comm_name is None:
            continue
        shapes = (
            distributed_model.sequence_shapes
            if name == "pos_embed" and dim == 1
            else None
        )
        gradient = _gather(gradient, dim, shapes, comm_name)
    return gradient


class TestParallelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp_size = int(os.environ.get("TP_SIZE", "1" if world_size == 1 else "2"))
        cp_size = int(os.environ.get("CP_SIZE", str(world_size // tp_size)))
        if tp_size * cp_size != world_size:
            raise AssertionError("TP_SIZE * CP_SIZE must equal WORLD_SIZE")

        backend = os.environ.get(
            "BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            cls.device = torch.device("cuda", local_rank)
        else:
            cls.device = torch.device("cpu")
        comm.initialize(
            tp_size=tp_size,
            cp_size=cp_size,
            backend=backend,
        )
        cls.rank = comm.get_world_rank()
        if set(comm.get_names()) != {"tp", "cp"}:
            raise AssertionError("test requires exactly tp and cp groups")

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        comm.destroy()

    def test_forward_input_gradient_and_weight_gradients(self) -> None:
        torch.manual_seed(SEED)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        baseline = build_baseline(self.device)

        torch.manual_seed(SEED + 1)
        input_expected = torch.randn(
            1,
            CONFIG["channels"],
            CONFIG["height"],
            CONFIG["width"],
            device=self.device,
            requires_grad=True,
        )
        target = torch.randn(
            1, CONFIG["num_classes"], device=self.device
        )
        output_expected = baseline(input_expected)
        # Sum reduction is explicit; no batch/token mean scaling is hidden.
        loss_expected = (output_expected - target).square().sum()
        loss_expected.backward()
        input_gradient_expected = input_expected.grad.detach().clone()
        parameter_gradients_expected = {
            name: parameter.grad.detach().clone()
            for name, parameter in baseline.named_parameters()
        }

        distributed_model = build_distributed(self.device)
        validate_param_comm_metadata(distributed_model)
        copy_logical_parameters(baseline, distributed_model)
        init_params_for_shared_weights(distributed_model)
        handles = register_model_parallel_gradient_hooks(distributed_model)

        input_actual = input_expected.detach().clone().requires_grad_(True)
        output_actual = distributed_model(input_actual)
        loss_actual = (output_actual - target).square().sum()
        loss_actual = distributed_model.scale_loss_for_context_parallel(
            loss_actual
        )
        loss_actual.backward()

        assert_tensor_close(
            self,
            "model output",
            output_expected.detach(),
            output_actual.detach(),
            self.rank,
        )
        if input_actual.grad is None:
            self.fail(f"input gradient missing on rank={self.rank}")
        assert_tensor_close(
            self,
            "input gradient",
            input_gradient_expected,
            input_actual.grad,
            self.rank,
        )

        self.assertEqual(
            parameter_gradients_expected.keys(),
            dict(distributed_model.named_parameters()).keys(),
        )
        for name, parameter in distributed_model.named_parameters():
            reconstructed = reconstruct_gradient(
                name, parameter, distributed_model
            )
            assert_tensor_close(
                self,
                f"parameter gradient {name}",
                parameter_gradients_expected[name],
                reconstructed,
                self.rank,
            )
        for handle in handles:
            handle.remove()

    def test_group_membership_matches_mesh_coordinates(self) -> None:
        tp_size = comm.get_size("tp")
        cp_size = comm.get_size("cp")
        world_rank = comm.get_world_rank()
        cp_coordinate, tp_coordinate = divmod(world_rank, tp_size)
        expected_tp = [
            cp_coordinate * tp_size + offset for offset in range(tp_size)
        ]
        expected_cp = [
            coordinate * tp_size + tp_coordinate
            for coordinate in range(cp_size)
        ]
        self.assertEqual(comm.get_ranks("tp"), expected_tp)
        self.assertEqual(comm.get_ranks("cp"), expected_cp)
        self.assertEqual(comm.get_root("tp"), min(expected_tp))
        self.assertEqual(comm.get_root("cp"), min(expected_cp))

    def test_shared_parameter_initialization(self) -> None:
        model = build_distributed(self.device)
        validate_param_comm_metadata(model)
        with torch.no_grad():
            model.head.weight.fill_(float(self.rank + 1))
        init_params_for_shared_weights(model)
        expected = torch.ones_like(model.head.weight)
        assert_tensor_close(
            self,
            "shared head initialization",
            expected,
            model.head.weight,
            self.rank,
        )

    @unittest.skipUnless(
        os.environ.get("RUN_PRODUCTION_SHAPE") == "1",
        "set RUN_PRODUCTION_SHAPE=1 for the 32-GPU smoke test",
    )
    def test_production_shape_smoke(self) -> None:
        self.assertEqual(
            (comm.get_size("tp"), comm.get_size("cp")),
            (4, 8),
            "production smoke requires tp=4, cp=8",
        )
        model = VisionTransformer().to(
            device=self.device, dtype=torch.bfloat16
        )
        validate_param_comm_metadata(model)
        init_params_for_shared_weights(model)
        handles = register_model_parallel_gradient_hooks(model)
        image = torch.randn(
            1,
            70,
            720,
            1440,
            device=self.device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        logits = model(image)
        loss = model.scale_loss_for_context_parallel(logits.float().sum())
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertIsNotNone(image.grad)
        if self.device.type == "cuda":
            self.assertLess(
                torch.cuda.max_memory_reserved(self.device),
                40 * 10**9,
                f"rank={self.rank} exceeded the 40GB memory target",
            )
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
