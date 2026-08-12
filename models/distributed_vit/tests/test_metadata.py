import unittest

from models.distributed_vit import comm
from models.distributed_vit.metadata import validate_param_comm_metadata
from models.distributed_vit.model import VisionTransformer


class TestMetadata(unittest.TestCase):
    def setUp(self) -> None:
        comm.initialize(tp_size=1, cp_size=1)

    def tearDown(self) -> None:
        comm.destroy()

    def test_every_trainable_parameter_has_valid_metadata(self) -> None:
        model = VisionTransformer(
            image_size=(8, 12),
            patch_size=2,
            in_channels=3,
            num_classes=7,
            dim=16,
            depth=2,
            heads=4,
            mlp_ratio=2.0,
        )
        validate_param_comm_metadata(model)
        for name, parameter in model.named_parameters():
            self.assertTrue(
                hasattr(parameter, "comm_metadata"),
                f"{name} is missing comm_metadata",
            )
            self.assertEqual(
                len(parameter.comm_metadata["sharded"]),
                parameter.ndim,
                f"{name} metadata does not match stored layout",
            )

    def test_unknown_group_fails_validation(self) -> None:
        model = VisionTransformer(
            image_size=(8, 12),
            patch_size=2,
            in_channels=3,
            num_classes=7,
            dim=16,
            depth=1,
            heads=4,
            mlp_ratio=2.0,
        )
        model.head.weight.comm_metadata["shared"].append("unknown")
        with self.assertRaisesRegex(ValueError, "unknown groups"):
            validate_param_comm_metadata(model)


if __name__ == "__main__":
    unittest.main()
