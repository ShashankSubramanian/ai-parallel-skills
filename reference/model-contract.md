# Model Contract

The contract refers to a string representation of the tensor shapes involved or mathematical operations. This reference formalizes how to write tensor and operation contracts for any model / layer.
Analyze any model / layer semantics using tensor and operation contracts listed below as much as possible and where possible.
Use the target project's layout when it already has one. Otherwise use the following conventions:

- `b`: batch size.
- `l`: sequence or context length.
- `h`: height of image or volume.
- `w`: width of image or volume.
- `d`: depth of volume.
- `a`: number of attention heads.
- `e_head`: per-head hidden dimension.
- `n`: number of layers.
- `v`: vocabulary size.
- `e`: input embedding or hidden dimension.
- `f`: output embedding or hidden dimension.


## Tensor Contract for Logical Tensors

Write logical tensor shapes as:
```text
b l e
```

## Model Operations Contract for Logical Tensors

Write matmuls / tensor ops using einsum convention as applicable:

```text
b l e, e f -> b l f
```

If operation are not directly matmuls, then use the actual operation contract. For example, a softmax operation is:
```text
softmax(b l e) -> b l e
```

or activation function is:

```text
relu(b l e) -> b l e
```

Some operations like convolutions may not have simple matmuls, you can fall back to the actual operation contract as well. For example, a convolution is:
```text
conv2d(b h w c_in, c_in c_out k_h k_w) -> b h' w' c_out
```