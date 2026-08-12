import unittest

from models.distributed_vit.rank_generator import (
    RankGenerator,
    generate_masked_orthogonal_rank_groups,
)


class TestRankGenerator(unittest.TestCase):
    def test_tp_cp_groups_match_node_layout(self) -> None:
        generator = RankGenerator({"tp": 4, "cp": 8}, order="tp-cp")
        self.assertEqual(
            generator.get_ranks("tp"),
            [list(range(start, start + 4)) for start in range(0, 32, 4)],
        )
        self.assertEqual(
            generator.get_ranks("cp"),
            [list(range(offset, 32, 4)) for offset in range(4)],
        )

    def test_combined_group_covers_world_once(self) -> None:
        generator = RankGenerator({"tp": 2, "cp": 2}, order="tp-cp")
        self.assertEqual(generator.get_ranks("tp-cp"), [[0, 1, 2, 3]])
        for token in ("tp", "cp", "tp-cp"):
            flattened = [
                rank for group in generator.get_ranks(token) for rank in group
            ]
            self.assertEqual(sorted(flattened), list(range(4)))

    def test_invalid_mesh_product_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "mesh product"):
            generate_masked_orthogonal_rank_groups(3, [2, 2], [True, False])

    def test_invalid_order_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            RankGenerator({"tp": 2, "cp": 2}, order="tp")


if __name__ == "__main__":
    unittest.main()
