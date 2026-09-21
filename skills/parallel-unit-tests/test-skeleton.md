# Distributed Equivalence Test Skeleton

Copy and adapt this structure to the target repository. Replace every
`PROJECT:` marker with the project's topology, mapping, metadata, model, and
loss APIs. Keep the lifecycle and assertions unless the parallelism plan
explains why they do not apply.

```python
import contextlib
import os
import unittest

import torch

# PROJECT: import comm, model, mappings, metadata validation, and loss APIs.


SEED = 333
RTOL = 1e-5
ATOL = 1e-6


def seed_every_rank() -> None:
    # Use the same seed on every rank for the logical oracle/input/target.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def diagnostic(name, expected, actual, *, rank, group="world") -> str:
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


def assert_tensor_close(test, name, expected, actual, *, rank, group="world"):
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


def named_parameter_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(f"missing gradient for trainable parameter {name}")
        if parameter.grad.shape != parameter.shape:
            raise AssertionError(
                f"{name}: grad_shape={tuple(parameter.grad.shape)} "
                f"parameter_shape={tuple(parameter.shape)}"
            )
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def slice_for_rank(tensor, dim, group_name, split_sizes=None):
    """Return this rank's shard without assuming all shards are equal."""
    # PROJECT: obtain group-local rank/size, not global rank/world size.
    group_rank = get_group_rank(group_name)
    group_size = get_group_size(group_name)

    if split_sizes is None:
        assert tensor.shape[dim] % group_size == 0, (
            f"dimension {dim} size {tensor.shape[dim]} is not divisible by "
            f"group {group_name} size {group_size}"
        )
        local_size = tensor.shape[dim] // group_size
        split_sizes = [local_size] * group_size
    assert len(split_sizes) == group_size
    assert sum(split_sizes) == tensor.shape[dim]

    start = sum(split_sizes[:group_rank])
    return tensor.narrow(dim, start, split_sizes[group_rank])


@torch.no_grad()
def copy_logical_parameters(baseline, distributed_model, split_sizes_by_key):
    """Copy each full baseline parameter into the exact distributed shard."""
    baseline_parameters = dict(baseline.named_parameters())
    distributed_parameters = dict(distributed_model.named_parameters())
    if baseline_parameters.keys() != distributed_parameters.keys():
        missing = baseline_parameters.keys() - distributed_parameters.keys()
        extra = distributed_parameters.keys() - baseline_parameters.keys()
        raise AssertionError(f"parameter-name mismatch: missing={missing}, extra={extra}")

    for name, local_parameter in distributed_parameters.items():
        if not hasattr(local_parameter, "comm_metadata"):
            raise AssertionError(f"{name}: missing comm_metadata")
        metadata = local_parameter.comm_metadata
        logical_shard = baseline_parameters[name].detach()

        # Contract: sharded[dim] is None or the group sharding tensor dim.
        for dim, group_name in enumerate(metadata["sharded"]):
            if group_name is None:
                continue
            # PROJECT: key split tables by parameter/group/dimension as needed.
            split_sizes = split_sizes_by_key.get((name, dim, group_name))
            logical_shard = slice_for_rank(
                logical_shard, dim, group_name, split_sizes
            )

        if logical_shard.shape != local_parameter.shape:
            raise AssertionError(
                f"{name}: copied shard shape={tuple(logical_shard.shape)} "
                f"local parameter shape={tuple(local_parameter.shape)} "
                f"metadata={metadata}"
            )
        local_parameter.copy_(logical_shard)


def reconstruct_parameter_gradient(name, parameter, split_sizes_by_key):
    gradient = parameter.grad.detach()
    for dim, group_name in enumerate(parameter.comm_metadata["sharded"]):
        if group_name is None:
            continue
        split_sizes = split_sizes_by_key.get((name, dim, group_name))
        # PROJECT: gather on group_name and concatenate on dim. The helper must
        # support split_sizes when local shards are uneven.
        gradient = gather_along_dim(
            gradient,
            dim=dim,
            group_name=group_name,
            split_sizes=split_sizes,
        )
    return gradient


@contextlib.contextmanager
def baseline_communication_disabled():
    """Prevent constructors from making the baseline model-parallel."""
    # PROJECT: temporarily hide only the model-parallel group registry or pass
    # an explicit non-distributed configuration. Preserve the initialized
    # process group. Restore state in finally.
    old_state = disable_model_parallel_lookup()
    try:
        yield
    finally:
        restore_model_parallel_lookup(old_state)


class TestParallelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # PROJECT: comm.py owns device selection and all distributed setup.
        backend = os.getenv(
            "BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        comm.initialize(backend=backend)
        cls.rank = comm.get_world_rank()
        cls.world_size = comm.get_world_size()
        cls.local_rank = comm.get_local_rank()
        cls.device = torch.device(
            f"cuda:{cls.local_rank}" if backend == "nccl" else "cpu"
        )

        # PROJECT: assert mesh product/world-size and required group names.
        validate_topology()
        seed_every_rank()

    @classmethod
    def tearDownClass(cls):
        comm.destroy()

    def test_forward_and_backward_equivalence(self):
        # Keep a tiny asymmetric case. Parameterize additional mesh/shape cases.
        config = {
            "batch": 4,
            "height": 12,
            "width": 20,
            "channels": 3,
        }

        # ----- Full logical baseline -----
        full_metadata = make_full_metadata(config)  # PROJECT
        with baseline_communication_disabled():
            baseline = build_model(config, full_metadata).to(self.device)  # PROJECT
        baseline.zero_grad(set_to_none=True)

        input_full = torch.randn(
            config["batch"],
            config["channels"],
            config["height"],
            config["width"],
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        output_expected = baseline(input_full)
        target_full = torch.randn_like(output_expected)
        baseline_loss = build_loss(full_metadata).to(self.device)  # PROJECT
        loss_expected = baseline_loss(output_expected, target_full)
        self.assertEqual(loss_expected.numel(), 1, "baseline loss must be scalar")
        loss_expected.backward()

        output_expected = output_expected.detach().clone()
        input_gradient_expected = input_full.grad.detach().clone()
        parameter_gradients_expected = named_parameter_gradients(baseline)

        # ----- Distributed path -----
        # PROJECT: derive exact local sizes after patch/window/padding rules.
        split_plan = compute_split_plan(config)
        local_metadata = split_metadata(full_metadata, split_plan)
        distributed_model = build_model(config, local_metadata).to(self.device)
        distributed_model.zero_grad(set_to_none=True)

        validate_model_groups(distributed_model)  # PROJECT
        annotate_and_validate_comm_metadata(distributed_model)  # PROJECT

        copy_logical_parameters(
            baseline,
            distributed_model,
            split_plan.parameter_split_sizes,
        )
        synchronize_shared_parameters(distributed_model)  # PROJECT, if required
        assert_shared_parameters_equal(distributed_model)  # PROJECT

        distributed_model = install_distributed_wrappers_and_hooks(
            distributed_model
        )
        local_module = getattr(distributed_model, "module", distributed_model)

        # Split from the same full tensors. Make input_local a leaf so .grad is
        # directly available after backward.
        with torch.no_grad():
            input_local = split_input(input_full.detach(), split_plan)  # PROJECT
            target_local = split_target(target_full.detach(), split_plan)  # PROJECT
        input_local = input_local.detach().requires_grad_(True)

        output_local = distributed_model(input_local)
        distributed_loss = build_loss(local_metadata).to(self.device)
        loss_local = distributed_loss(output_local, target_local)
        self.assertEqual(loss_local.numel(), 1, "distributed loss must be scalar")

        # PROJECT: make loss_local mathematically equivalent to loss_expected.
        # Derive any factor from loss reductions and DDP/hook semantics.
        loss_local = scale_local_loss_if_required(loss_local, split_plan)
        loss_local.backward()

        self.assertIsNotNone(input_local.grad, "local input gradient is missing")
        parameter_gradients_local = named_parameter_gradients(local_module)

        # ----- Reconstruct and compare -----
        with torch.no_grad():
            output_actual = gather_output(output_local, split_plan)  # PROJECT
            input_gradient_actual = gather_input_gradient(
                input_local.grad, split_plan
            )  # PROJECT
            input_gradient_actual = correct_input_gradient_scale(
                input_gradient_actual, split_plan
            )  # PROJECT

        assert_tensor_close(
            self,
            "model output",
            output_expected,
            output_actual,
            rank=self.rank,
        )
        assert_tensor_close(
            self,
            "input gradient",
            input_gradient_expected,
            input_gradient_actual,
            rank=self.rank,
        )

        self.assertEqual(
            parameter_gradients_expected.keys(),
            parameter_gradients_local.keys(),
        )
        for name, parameter in local_module.named_parameters():
            if not parameter.requires_grad:
                continue
            reconstructed = reconstruct_parameter_gradient(
                name,
                parameter,
                split_plan.parameter_split_sizes,
            )
            assert_tensor_close(
                self,
                f"parameter gradient {name}",
                parameter_gradients_expected[name],
                reconstructed,
                rank=self.rank,
            )


if __name__ == "__main__":
    unittest.main()
```

## Required Adaptations

Before considering the test complete:

- Replace all `PROJECT:` APIs and comments.
- State the logical and local shapes beside the split/gather code.
- State whether each loss dimension uses sum or mean reduction.
- State whether DDP/hooks sum or average each gradient.
- Record uneven split sizes and use the same tables for parameter copying,
  input/target splitting, output gathering, and gradient reconstruction.
- Ensure parameter annotation happens before code that consumes metadata.
- Confirm wrapper prefixes such as `module.` do not change parameter-name
  matching.
- Add the exact local and scheduler launch commands.
- Add a world-size-one case and at least one multi-axis case.

The skeleton intentionally compares named parameter gradients rather than only
concatenating them. An optional flattened relative-error summary may be added
after all per-parameter checks pass.
