# Example Parallelism

Use these examples as common patterns to use or guide the planning, not as mandatory implementations. Always adapt axis names to the target model.

## MLP

Logical:

```text
fc1: b l e, e f -> b l f
act: b l f -> b l f
fc2: b l f, f e -> b l e
```

Data parallel:

```text
fc1: b/dp l e, e f -> b/dp l f
act: b/dp l f -> b/dp l f
fc2: b/dp l f, f e -> b/dp l e
wgrads: all-reduce over dp since weights are shared across `dp`.
```

Tensor parallel, paired column then row:

```text
fc1: b l e, e f/tp -> b l f/tp
act: b l f/tp -> b l f/tp
fc2: b l f/tp, f/tp e -> b l e
fc2 output: all-reduce over tp
```

Context parallel:

```text
fc1: b l/cp e, e f -> b l/cp f
fc2: b l/cp f, f e -> b l/cp e
weight gradients: all-reduce over cp
```

## Attention

Logical:

```text
qkv: b l e, e 3a d -> b l 3a d
scores: b l a d, b m a d -> b a l m
probs: softmax over m
ctx: b a l m, b m a d -> b l a d
out: b l a d, a d e -> b l e
```

Data parallel:

```text
qkv: b/dp l e, e 3a d -> b/dp l 3a d
scores: b/dp l a d, b/dp m a d -> b/dp a l m
ctx: b/dp a l m, b/dp m a d -> b/dp l a d
gradients: all-reduce over dp
```

Tensor parallel over heads:

```text
qkv: b l e, e 3a/tp d -> b l 3a/tp d
scores: b l a/tp d, b m a/tp d -> b a/tp l m
ctx: b a/tp l m, b m a/tp d -> b l a/tp d
out: b l a/tp d, a/tp d e -> b l e
out: all-reduce over tp
```

Context parallel over sequence:

```text
qkv: b l/cp e, e 3a d -> b l/cp 3a d
scores: b l/cp a d, b m a d -> b a l/cp m
ctx: b a l/cp m, b m a d -> b l/cp a d
keys/values: gather or stream over cp depending on attention implementation
```

## Conv2d

Logical:

```text
conv2d: b h w c_in, c_out c_in k_h k_w -> b h w c_out
```

Data parallel:

```text
conv2d: b/dp h w c_in, c_out c_in k_h k_w -> b/dp h w c_out
gradients: all-reduce over dp
```

Tensor parallel over output channels:

```text
conv2d: b h w c_in, c_out/tp c_in k_h k_w -> b h w c_out/tp
```

Tensor parallel over input channels:

```text
conv2d: b h w c_in/tp, c_out c_in/tp k_h k_w -> b h w c_out
output: all-reduce over tp
```

Spatial parallel:

```text
conv2d: b h/sp_h w/sp_w c_in, c_out c_in k_h k_w -> b h/sp_h w/sp_w c_out
halo: exchange h and w boundary regions before local conv
```
