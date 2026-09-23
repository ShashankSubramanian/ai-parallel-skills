"""Complete context- and tensor-parallel Vision Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import comm
from .mappings import (
    allgather_forward_reducescatter_backward,
    allreduce_forward_identity_backward,
    identity_forward_allreduce_backward,
)
from .metadata import set_comm_metadata


class ColumnParallelLinear(nn.Module):
    """Linear with output features sharded over TP."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        tp = comm.get_size("tp")
        if out_features % tp:
            raise ValueError(f"out_features={out_features} is not divisible by tp={tp}")
        self.in_features = in_features
        self.out_features = out_features
        self.local_out_features = out_features // tp
        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = (
            nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = identity_forward_allreduce_backward(x, "tp")
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear with input features sharded over TP and full reduced output."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        tp = comm.get_size("tp")
        if in_features % tp:
            raise ValueError(f"in_features={in_features} is not divisible by tp={tp}")
        self.in_features = in_features
        self.out_features = out_features
        self.local_in_features = in_features // tp
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight, None)
        output = allreduce_forward_identity_backward(partial, "tp")
        if self.bias is not None:
            output = output + self.bias
        return output


class DistributedAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        tp = comm.get_size("tp")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if heads % tp:
            raise ValueError(f"heads={heads} must be divisible by tp={tp}")
        if dropout != 0.0:
            raise ValueError("the approved distributed configuration requires dropout=0")
        self.heads = heads
        self.local_heads = heads // tp
        self.head_dim = dim // heads
        self.qkv = ColumnParallelLinear(dim, dim * 3)
        self.proj = RowParallelLinear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, real_tokens: int) -> torch.Tensor:
        batch, local_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            local_tokens,
            3,
            self.local_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_local, v_local = qkv.unbind(dim=0)
        local_shapes = [local_tokens] * comm.get_size("cp")
        k = allgather_forward_reducescatter_backward(
            k_local, 2, local_shapes, "cp"
        )
        v = allgather_forward_reducescatter_backward(
            v_local, 2, local_shapes, "cp"
        )
        # Padding is appended globally, so slicing K/V removes every dummy key.
        k = k[:, :, :real_tokens, :]
        v = v[:, :, :real_tokens, :]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(
            batch, local_tokens, dim // comm.get_size("tp")
        )
        return self.proj(output)


class DistributedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dropout != 0.0:
            raise ValueError("the approved distributed configuration requires dropout=0")
        self.net = nn.Sequential(
            ColumnParallelLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            RowParallelLinear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistributedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DistributedAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DistributedMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, real_tokens: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), real_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class DistributedVisionTransformer(nn.Module):
    """ViT with CP token shards and TP attention/MLP projections."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        *,
        sequence_padding_multiple: int = 32,
    ) -> None:
        super().__init__()
        if isinstance(image_size, int):
            image_height, image_width = image_size, image_size
        else:
            image_height, image_width = image_size
        if image_height % patch_size or image_width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if sequence_padding_multiple < 1:
            raise ValueError("sequence_padding_multiple must be positive")
        cp = comm.get_size("cp")
        if sequence_padding_multiple % cp:
            raise ValueError(
                f"sequence_padding_multiple={sequence_padding_multiple} must be "
                f"divisible by cp={cp}"
            )

        patch_count = (image_height // patch_size) * (image_width // patch_size)
        self.real_tokens = patch_count + 1
        self.padded_tokens = (
            math.ceil(self.real_tokens / sequence_padding_multiple)
            * sequence_padding_multiple
        )
        if self.padded_tokens % cp:
            raise ValueError("padded sequence length must be divisible by cp")
        self.local_tokens = self.padded_tokens // cp
        self.image_size = (image_height, image_width)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.heads = heads

        self.patch_embed = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.local_tokens, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DistributedBlock(dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()
        self._annotate_parameters()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        local_start = comm.get_rank("cp") * self.local_tokens
        valid_local = max(0, min(self.local_tokens, self.real_tokens - local_start))
        if valid_local < self.local_tokens:
            with torch.no_grad():
                self.pos_embed[:, valid_local:].zero_()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _annotate_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            sharded: list[str | None] = [None] * parameter.ndim
            shared: list[str] = ["tp", "cp"]
            reduce: list[str] = ["cp"]

            if name == "pos_embed":
                sharded = [None, "cp", None]
                shared = ["tp"]
                reduce = []
            elif name.endswith("attn.qkv.weight"):
                sharded = ["tp", None]
                shared = ["cp"]
            elif name.endswith("attn.qkv.bias"):
                sharded = ["tp"]
                shared = ["cp"]
            elif name.endswith("attn.proj.weight"):
                sharded = [None, "tp"]
                shared = ["cp"]
            elif name.endswith("mlp.net.0.weight"):
                sharded = ["tp", None]
                shared = ["cp"]
            elif name.endswith("mlp.net.0.bias"):
                sharded = ["tp"]
                shared = ["cp"]
            elif name.endswith("mlp.net.3.weight"):
                sharded = [None, "tp"]
                shared = ["cp"]
            elif name in {"cls_token", "patch_embed.weight", "patch_embed.bias"}:
                pass
            elif (
                ".norm1." in name
                or ".norm2." in name
                or name.startswith("norm.")
                or name.startswith("head.")
                or name.endswith("attn.proj.bias")
                or name.endswith("mlp.net.3.bias")
            ):
                pass
            else:
                raise ValueError(f"no approved metadata rule for parameter {name!r}")

            set_comm_metadata(
                parameter,
                sharded=sharded,
                shared=shared,
                reduce=reduce,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, *self.image_size)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected input shape (batch,{expected}), got {tuple(x.shape)}"
            )
        # Each CP rank computes a token slice. Sum their partial input gradients.
        x = identity_forward_allreduce_backward(x, "cp")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        if self.padded_tokens > self.real_tokens:
            padding = x.new_zeros(
                x.shape[0],
                self.padded_tokens - self.real_tokens,
                self.dim,
            )
            x = torch.cat((x, padding), dim=1)
        start = comm.get_rank("cp") * self.local_tokens
        x = x.narrow(1, start, self.local_tokens).contiguous()
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, self.real_tokens)
        x = self.norm(x)

        # CP rank 0 owns CLS. Calling the head on every rank keeps every
        # parameter in the graph; non-owner ranks contribute exact zeros.
        owner = 1.0 if comm.get_rank("cp") == 0 else 0.0
        return self.head(x[:, 0]) * owner
