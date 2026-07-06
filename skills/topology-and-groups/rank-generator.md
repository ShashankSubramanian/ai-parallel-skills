# Rank Generator

Use this subskill to create the pure rank-math building block for topology work.
Copy or adapt Megatron's `generate_masked_orthogonal_rank_groups` and
`RankGenerator` pattern.

## Required API

Implement a side-effect-free module, usually `rank_generator.py`, with:

- `generate_masked_orthogonal_rank_groups(world_size, parallel_size, mask)`.
- `RankGenerator(..., order: str)` or an equivalent class accepting dimension
  names, sizes, and order.
- `RankGenerator.get_ranks(token: str) -> list[list[int]]`, where `token` can be
  a single dimension (`tp`) or a hyphen-joined product (`tp-cp`, `sp1-sp2`).

## Required Behavior

- Validate `world_size == product(parallel_size)`.
- Validate every non-size-one dimension appears in `order`.
- Preserve the project rank order exactly; do not silently reorder dimensions.
- Generate orthogonal hyperrectangle groups by mask:
  - Masked dimensions vary inside each group.
  - Unmasked dimensions choose which group a rank belongs to.
- Support arbitrary planned dimension names such as `dp`, `tp`, `cp`, `sp1`,
  `sp2`, and `ep` when needed.
- Keep this module independent of `torch.distributed`.

## Minimal Example

```python
rank_gen = RankGenerator(
    dimensions={"sp1": 2, "sp2": 2, "tp": 4, "dp": 8},
    order="sp1-sp2-tp-dp",
)

tp_groups = rank_gen.get_ranks("tp")
sp_groups = rank_gen.get_ranks("sp1-sp2")
model_groups = rank_gen.get_ranks("tp-sp1-sp2")
```

## Tests

- For every requested token, every rank appears in exactly one group.
- Every group has size `product(token_dims)`.
- Number of groups is `world_size // product(token_dims)`.
- Known small meshes match hand-written expected ranks.

## Reference

- Megatron-LM `parallel_state.py`: `generate_masked_orthogonal_rank_groups` and
  `RankGenerator`.
