"""Distributed topology, metadata, and ViT equivalence tests."""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist

from vit import VisionTransformer

from . import comm
from .helpers import _gather, _reduce
from .mappings import init_ddp_model_and_reduction_hooks
from .metadata import init_params_for_shared_weights, validate_param_comm_metadata
from .model import DistributedVisionTransformer
from .rank_generator import RankGenerator

SEED = 333
RTOL = 2e-4
ATOL = 2e-5


def seed_every_rank() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def diagnostic(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rank: int,
    group: str = "world",
) -> str:
    if expected.shape != actual.shape:
        return (
            f"{name}: rank={rank} group={group} "
            f"expected_shape={tuple(expected.shape)} "
            f"actual_shape={tuple(actual.shape)}"
        )
    difference = actual.detach().float() - expected.detach().float()
    denominator = expected.detach().float().norm().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (
        f"{name}: rank={rank} group={group} shape={tuple(actual.shape)} "
        f"dtype={actual.dtype} max_abs={difference.abs().max().item():.6e} "
        f"rel_l2={(difference.norm() / denominator).item():.6e} "
        f"rtol={RTOL} atol={ATOL}"
    )


def assert_tensor_close(
    test: unittest.TestCase,
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rank: int,
    group: str = "world",
) -> None:
    test.assertEqual(
        tuple(actual.shape),
        tuple(expected.shape),
        diagnostic(name, expected, actual, rank=rank, group=group),
    )
    test.assertTrue(
        torch.isfinite(expected).all().item(),
        f"{name}: non-finite baseline tensor on rank={rank}",
    )
    test.assertTrue(
        torch.isfinite(actual).all().item(),
        f"{name}: non-finite distributed tensor on rank={rank}",
    )
    try:
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError as error:
        raise AssertionError(
            diagnostic(name, expected, actual, rank=rank, group=group)
        ) from error


def mesh_from_environment() -> dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    default_tp = 1 if world_size == 1 else int(os.environ.get("TP_SIZE", "4"))
    if world_size % default_tp:
        raise ValueError(f"WORLD_SIZE={world_size} is not divisible by TP_SIZE")
    default_cp = world_size // default_tp
    dimensions = {
        "dp": int(os.environ.get("DP_SIZE", "1")),
        "tp": default_tp,
        "cp": int(os.environ.get("CP_SIZE", str(default_cp))),
    }
    if dimensions["dp"] * dimensions["tp"] * dimensions["cp"] != world_size:
        raise ValueError(
            f"test mesh {dimensions} does not multiply to WORLD_SIZE={world_size}"
        )
    return dimensions


def _qkv_local_shard(
    tensor: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    tp = comm.get_size("tp")
    tp_rank = comm.get_rank("tp")
    local_heads = heads // tp
    if tensor.ndim == 2:
        view = tensor.view(3, heads, head_dim, tensor.shape[1])
        shard = view[
            :,
            tp_rank * local_heads : (tp_rank + 1) * local_heads,
            :,
            :,
        ]
    else:
        view = tensor.view(3, heads, head_dim)
        shard = view[
            :,
            tp_rank * local_heads : (tp_rank + 1) * local_heads,
            :,
        ]
    return shard.reshape(-1, *tensor.shape[1:]).contiguous()


@torch.no_grad()
def copy_logical_parameters(
    baseline: VisionTransformer,
    distributed_model: DistributedVisionTransformer,
) -> None:
    baseline_parameters = dict(baseline.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    if baseline_parameters.keys() != distributed_parameters.keys():
        missing = baseline_parameters.keys() - distributed_parameters.keys()
        extra = distributed_parameters.keys() - baseline_parameters.keys()
        raise AssertionError(
            f"parameter-name mismatch: missing={missing}, extra={extra}"
        )

    for name, local_parameter in distributed_parameters.items():
        logical = baseline_parameters[name].detach()
        if name == "pos_embed":
            padded = logical.new_zeros(
                1, distributed_model.padded_tokens, logical.shape[-1]
            )
            padded[:, : distributed_model.real_tokens].copy_(logical)
            start = comm.get_rank("cp") * distributed_model.local_tokens
            logical = padded.narrow(1, start, distributed_model.local_tokens)
        elif name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias"):
            logical = _qkv_local_shard(
                logical,
                heads=distributed_model.heads,
                head_dim=distributed_model.dim // distributed_model.heads,
            )
        else:
            for dim, group_name in enumerate(
                local_parameter.comm_metadata["sharded"]
            ):
                if group_name is None:
                    continue
                group_size = comm.get_size(group_name)
                if logical.shape[dim] % group_size:
                    raise AssertionError(
                        f"{name}: dimension {dim} is not divisible by {group_name}"
                    )
                chunk = logical.shape[dim] // group_size
                logical = logical.narrow(
                    dim, comm.get_rank(group_name) * chunk, chunk
                )
        if logical.shape != local_parameter.shape:
            raise AssertionError(
                f"{name}: rank={comm.get_world_rank()} copied_shape="
                f"{tuple(logical.shape)} local_shape={tuple(local_parameter.shape)} "
                f"metadata={local_parameter.comm_metadata}"
            )
        local_parameter.copy_(logical)


def _reconstruct_qkv_gradient(
    gradient: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    gathered = _gather(gradient, 0, None, "tp")
    tp = comm.get_size("tp")
    local_heads = heads // tp
    if gradient.ndim == 2:
        gathered = gathered.view(tp, 3, local_heads, head_dim, gradient.shape[1])
        return gathered.permute(1, 0, 2, 3, 4).reshape(
            3 * heads * head_dim, gradient.shape[1]
        )
    gathered = gathered.view(tp, 3, local_heads, head_dim)
    return gathered.permute(1, 0, 2, 3).reshape(3 * heads * head_dim)


def reconstruct_parameter_gradient(
    name: str,
    parameter: torch.nn.Parameter,
    model: DistributedVisionTransformer,
) -> torch.Tensor:
    if parameter.grad is None:
        raise AssertionError(
            f"{name}: missing gradient on rank={comm.get_world_rank()}"
        )
    gradient = parameter.grad.detach()
    if name == "pos_embed":
        gradient = _gather(gradient, 1, None, "cp")
        return gradient[:, : model.real_tokens]
    if name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias"):
        return _reconstruct_qkv_gradient(
            gradient,
            heads=model.heads,
            head_dim=model.dim // model.heads,
        )
    for dim, group_name in enumerate(parameter.comm_metadata["sharded"]):
        if group_name is not None:
            gradient = _gather(gradient, dim, None, group_name)
    return gradient


@torch.no_grad()
def assert_shared_parameters_equal(
    test: unittest.TestCase, model: torch.nn.Module
) -> None:
    for name, parameter in model.named_parameters():
        for group_name in parameter.comm_metadata["shared"]:
            if comm.get_size(group_name) == 1:
                continue
            gathered = [torch.empty_like(parameter) for _ in range(comm.get_size(group_name))]
            dist.all_gather(
                gathered,
                parameter.contiguous(),
                group=comm.get_group(group_name),
            )
            for replica in gathered[1:]:
                assert_tensor_close(
                    test,
                    f"shared parameter {name}",
                    gathered[0],
                    replica,
                    rank=comm.get_world_rank(),
                    group=group_name,
                )


class TestRankGenerator(unittest.TestCase):
    def test_known_tp_cp_mesh(self) -> None:
        generator = RankGenerator(
            {"dp": 1, "tp": 2, "cp": 2}, order="tp-cp-dp"
        )
        self.assertEqual(generator.get_ranks("tp"), [[0, 1], [2, 3]])
        self.assertEqual(generator.get_ranks("cp"), [[0, 2], [1, 3]])
        self.assertEqual(
            generator.get_ranks("dp"), [[0], [1], [2], [3]]
        )
        for token in ("dp", "tp", "cp", "dp-tp", "dp-cp"):
            groups = generator.get_ranks(token)
            flattened = sorted(rank for group in groups for rank in group)
            self.assertEqual(flattened, [0, 1, 2, 3])


class TestParallelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dimensions = mesh_from_environment()
        backend = os.environ.get(
            "BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        comm.initialize(dimensions=cls.dimensions, backend=backend)
        cls.rank = comm.get_world_rank()
        cls.device = torch.device(
            f"cuda:{comm.get_local_rank()}" if backend == "nccl" else "cpu"
        )
        seed_every_rank()

    @classmethod
    def tearDownClass(cls) -> None:
        comm.destroy()

    def test_topology_accessors(self) -> None:
        self.assertEqual(
            comm.get_world_size(),
            self.dimensions["dp"]
            * self.dimensions["tp"]
            * self.dimensions["cp"],
        )
        for name in ("dp", "tp", "cp", "dp-tp", "dp-cp"):
            self.assertTrue(comm.is_initialized(name))
            self.assertIn(comm.get_world_rank(), comm.get_ranks(name))
            self.assertEqual(comm.get_size(name), len(comm.get_ranks(name)))
            self.assertEqual(comm.get_root(name), min(comm.get_ranks(name)))

    def test_forward_input_and_weight_gradient_equivalence(self) -> None:
        tp = self.dimensions["tp"]
        cp = self.dimensions["cp"]
        dim = max(8, 4 * tp)
        heads = max(tp, 2 * tp)
        image_size = (4, 2 * cp)
        patch_size = 2
        num_classes = 3

        seed_every_rank()
        # The immutable baseline never consults distributed state, so it stays
        # fully logical even while project process groups are initialized.
        baseline = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            dim=dim,
            depth=2,
            heads=heads,
            mlp_ratio=2.0,
            dropout=0.0,
        ).to(self.device)
        input_expected = torch.randn(
            1,
            3,
            *image_size,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        output_expected = baseline(input_expected)
        loss_expected = output_expected.square().sum()
        loss_expected.backward()
        output_expected = output_expected.detach().clone()
        input_gradient_expected = input_expected.grad.detach().clone()
        parameter_gradients_expected = {
            name: parameter.grad.detach().clone()
            for name, parameter in baseline.named_parameters()
        }

        distributed_model = DistributedVisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            dim=dim,
            depth=2,
            heads=heads,
            mlp_ratio=2.0,
            dropout=0.0,
            sequence_padding_multiple=cp,
        ).to(self.device)
        validate_param_comm_metadata(distributed_model)
        copy_logical_parameters(baseline, distributed_model)
        init_params_for_shared_weights(distributed_model)
        assert_shared_parameters_equal(self, distributed_model)
        wrapped = init_ddp_model_and_reduction_hooks(distributed_model)
        local_model = getattr(wrapped, "module", wrapped)

        # Model inputs are replicated by the approved plan. CP begins only
        # after patchification, so the local input shape equals the logical one.
        input_actual = input_expected.detach().clone().requires_grad_(True)
        output_local = wrapped(input_actual)
        # The scalar loss is a sum. CP rank 0 owns CLS and the other ranks keep
        # a graph-connected zero; CP gradient hooks sum parameter partials.
        if comm.get_rank("cp") == 0:
            loss_actual = output_local.square().sum()
        else:
            loss_actual = output_local.sum() * 0.0
        loss_actual.backward()
        output_actual = _reduce(output_local.detach(), "cp")

        assert_tensor_close(
            self,
            "model output",
            output_expected,
            output_actual,
            rank=self.rank,
        )
        if input_actual.grad is None:
            self.fail(f"input gradient missing on rank={self.rank}")
        assert_tensor_close(
            self,
            "input gradient",
            input_gradient_expected,
            input_actual.grad,
            rank=self.rank,
        )

        self.assertEqual(
            parameter_gradients_expected.keys(),
            dict(local_model.named_parameters()).keys(),
        )
        for name, parameter in local_model.named_parameters():
            reconstructed = reconstruct_parameter_gradient(
                name, parameter, local_model
            )
            assert_tensor_close(
                self,
                f"parameter gradient {name}",
                parameter_gradients_expected[name],
                reconstructed,
                rank=self.rank,
            )
        padded_gradient = local_model.pos_embed.grad
        self.assertIsNotNone(padded_gradient)
        local_start = comm.get_rank("cp") * local_model.local_tokens
        valid = max(
            0, min(local_model.local_tokens, local_model.real_tokens - local_start)
        )
        if valid < local_model.local_tokens:
            self.assertEqual(
                torch.count_nonzero(padded_gradient[:, valid:]).item(),
                0,
                f"padded position gradient is nonzero on rank={self.rank}",
            )

    @unittest.skipUnless(
        os.environ.get("RUN_REALISTIC_SHAPE") == "1",
        "set RUN_REALISTIC_SHAPE=1 for the realistic layout check",
    )
    def test_realistic_shape_contract(self) -> None:
        self.assertEqual((720 // 2) * (1440 // 2) + 1, 259201)
        self.assertEqual(259232 % self.dimensions["cp"], 0)
        self.assertEqual(16 % self.dimensions["tp"], 0)
        self.assertEqual(1024 % self.dimensions["tp"], 0)
        self.assertEqual(4096 % self.dimensions["tp"], 0)


if __name__ == "__main__":
    unittest.main()
