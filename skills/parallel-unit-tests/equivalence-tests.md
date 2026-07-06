# Equivalence Tests

Distributed correctness tests should compare logical tensors, not only local
rank shards.

## Forward Equivalence

1. Run the baseline model on one device with the full logical input.
2. Run the distributed model on sharded inputs or replicated inputs according to
   the plan.
3. Gather or reduce distributed outputs into the baseline logical shape.
4. Compare values and shapes.

## Input-Gradient Equivalence

1. Use the same scalar loss for baseline and distributed outputs.
2. Backpropagate once.
3. Gather or reduce input gradients into the baseline logical shape.
4. Compare gradients for every differentiable input.

## Weight-Gradient Equivalence

1. Start from identical logical weights.
2. After backward, compare each distributed weight gradient to the matching
   baseline slice.
3. If the implementation reduces gradients into replicated form, compare the
   full logical gradient on every relevant rank.
4. Assert that all trainable local parameters have gradients.

## Test Data

- Prefer small dimensions that are divisible by the chosen parallel sizes.
- Include at least one non-square or asymmetric shape to catch wrong-axis bugs.
- Use deterministic random inputs unless the model requires structured inputs.

## Failure Diagnostics

Report:

- Rank and group.
- Tensor name.
- Logical shape and local shape.
- Max absolute error and max relative error.
- The collective or mapping boundary most likely involved.
