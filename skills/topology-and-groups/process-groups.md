# Process Groups

Process groups should encode the sharding plan directly. Build them from the
Megatron-style hyperrectangle rank generator, then access them only through
`comm.py`.

## Group Design Checklist

- Define dimension names, sizes, and rank order before creating groups.
- Validate `world_size == product(active_dims)`.
- Build groups by token with `RankGenerator.get_ranks(token)`.
- Build one process group per rank list returned for each planned token.
- Cache the current rank's group handle and global rank list in `comm.py`.
- Add a debug print or structured summary that can be enabled during bring-up.

## Common Meshes

Data parallel only:

```text
mesh = [dp]
```

Tensor parallel with data parallel:

```text
mesh = [dp, tp]
```

Context parallel with data parallel:

```text
mesh = [dp, cp]
```

Spatial parallel with data parallel:

```text
mesh = [dp, sp]
```

Composed model parallel:

```text
mesh = [dp, tp, cp]
mesh = [dp, tp, sp]
mesh = [dp, tp, sp1, sp2]
```

## Common Group Tokens

Use tokens that match the approved plan and tensor metadata:

```text
dp
tp
cp
sp1
sp2
sp1-sp2
tp-cp
tp-sp1-sp2
dp-cp
dp-sp1-sp2
```

## Invariants To Test

- Every rank belongs to exactly one group for each mesh slice.
- Each group has the expected size.
- All ranks agree on group membership.
- Single-rank dimensions behave as identity dimensions.
- Sharded tensor dimensions are divisible by the relevant group size unless
  padding is explicitly implemented.
- `comm.get_group(token)`, `comm.get_size(token)`, `comm.get_rank(token)`, and
  `comm.get_root(token)` agree with the generated rank list for the current
  rank.
