"""Tensor- and context-parallel Vision Transformer."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from . import comm
from .distributed.helpers import compute_split_shapes
from .distributed.mappings import (
    allgather_forward_reducescatter_backward,
    allgather_forward_split_backward,
    allreduce_forward_identity_backward,
    broadcast_forward_reduce_to_root_backward,
    identity_forward_allreduce_backward,
    split_forward_allgather_backward,
)
from .metadata import set_comm_metadata


def _annotate_replicated_norm(module: nn.LayerNorm) -> None:
    if module.weight is not None:
        set_comm_metadata(
            module.weight,
            sharded=[None],
            shared=["tp", "cp"],
            reduce=["cp"],
        )
    if module.bias is not None:
        set_comm_metadata(
            module.bias,
            sharded=[None],
            shared=["tp", "cp"],
            reduce=["cp"],
        )


class ColumnParallelLinear(nn.Module):
    """Linear layer sharded over stored output dimension."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        tp_size = comm.get_size("tp")
        if out_features % tp_size != 0:
            raise ValueError("out_features must be divisible by tensor parallel size")
        self.in_features = in_features
        self.out_features = out_features
        self.local_out_features = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        )
        self.reset_parameters()
        set_comm_metadata(
            self.weight,
            sharded=["tp", None],
            shared=["cp"],
            reduce=["cp"],
        )
        if self.bias is not None:
            set_comm_metadata(
                self.bias,
                sharded=["tp"],
                shared=["cp"],
                reduce=["cp"],
            )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logical: (..., e) @ (e, f/tp) -> (..., f/tp).
        x = identity_forward_allreduce_backward(x, "tp")
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear layer sharded over stored input dimension."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        tp_size = comm.get_size("tp")
        if in_features % tp_size != 0:
            raise ValueError("in_features must be divisible by tensor parallel size")
        self.in_features = in_features
        self.out_features = out_features
        self.local_in_features = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
        set_comm_metadata(
            self.weight,
            sharded=[None, "tp"],
            shared=["cp"],
            reduce=["cp"],
        )
        if self.bias is not None:
            set_comm_metadata(
                self.bias,
                sharded=[None],
                shared=["tp", "cp"],
                reduce=["cp"],
            )

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            # Use the logical fan-in so a gathered initialization has the same
            # scale as nn.Linear(in_features, out_features).
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logical: (..., e/tp) @ (e/tp, f) -> partial (..., f).
        partial = F.linear(x, self.weight, None)
        output = allreduce_forward_identity_backward(partial, "tp")
        return output if self.bias is None else output + self.bias


class QKVParallelLinear(nn.Module):
    """Fused QKV projection with an explicit TP-sharded head dimension."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        tp_size = comm.get_size("tp")
        if dim % heads or heads % tp_size:
            raise ValueError("dim and heads must be divisible for QKV sharding")
        self.dim = dim
        self.heads = heads
        self.local_heads = heads // tp_size
        self.head_dim = dim // heads
        # Stored layout is (QKV, heads/tp, head_dim, input), so metadata
        # describes the actual head shard without a block-cyclic exception.
        self.weight = nn.Parameter(
            torch.empty(3, self.local_heads, self.head_dim, dim)
        )
        self.bias = nn.Parameter(
            torch.empty(3, self.local_heads, self.head_dim)
        )
        self.reset_parameters()
        set_comm_metadata(
            self.weight,
            sharded=[None, "tp", None, None],
            shared=["cp"],
            reduce=["cp"],
        )
        set_comm_metadata(
            self.bias,
            sharded=[None, "tp", None],
            shared=["cp"],
            reduce=["cp"],
        )

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = identity_forward_allreduce_backward(x, "tp")
        return F.linear(
            x,
            self.weight.reshape(-1, self.dim),
            self.bias.reshape(-1),
        )


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 16, dropout: float = 0.0
    ) -> None:
        super().__init__()
        tp_size = comm.get_size("tp")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if heads % tp_size != 0:
            raise ValueError("heads must be divisible by tensor parallel size")
        self.heads = heads
        self.local_heads = heads // tp_size
        self.head_dim = dim // heads
        self.qkv = QKVParallelLinear(dim, heads)
        self.proj = RowParallelLinear(dim, dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, sequence_shapes: Sequence[int]
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.local_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # Q remains context-local. K/V become exact global logical tensors;
        # uneven local lengths are padded only inside the collective helper.
        k = allgather_forward_reducescatter_backward(
            k, 2, sequence_shapes, "cp"
        )
        v = allgather_forward_reducescatter_backward(
            v, 2, sequence_shapes, "cp"
        )
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(
            batch, tokens, self.local_heads * self.head_dim
        )
        return self.proj(context)


class MLP(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ColumnParallelLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            RowParallelLinear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        _annotate_replicated_norm(self.norm1)
        _annotate_replicated_norm(self.norm2)

    def forward(
        self, x: torch.Tensor, sequence_shapes: Sequence[int]
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sequence_shapes)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Fixed rectangular ViT distributed over TP output features and CP tokens."""

    def __init__(
        self,
        image_size: tuple[int, int] = (720, 1440),
        patch_size: int = 2,
        in_channels: int = 70,
        num_classes: int = 1000,
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        height, width = image_size
        cp_size = comm.get_size("cp")
        tp_size = comm.get_size("tp")
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if (height // patch_size) % cp_size:
            raise ValueError("patch-grid height must be divisible by CP size")
        if dim % tp_size:
            raise ValueError("embedding dimension must be divisible by TP size")
        if dropout and tp_size * cp_size > 1:
            raise ValueError(
                "dropout must be zero for deterministic distributed semantics"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads

        self.input_height_shapes = compute_split_shapes(height, cp_size)
        patches_per_cp_rank = (
            (height // patch_size // cp_size) * (width // patch_size)
        )
        self.sequence_shapes = [
            patches_per_cp_rank + (1 if rank == 0 else 0)
            for rank in range(cp_size)
        ]
        cp_rank = comm.get_rank("cp")
        local_sequence = self.sequence_shapes[cp_rank]

        local_dim = dim // tp_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            local_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        set_comm_metadata(
            self.patch_embed.weight,
            sharded=["tp", None, None, None],
            shared=["cp"],
            reduce=["cp"],
        )
        if self.patch_embed.bias is not None:
            set_comm_metadata(
                self.patch_embed.bias,
                sharded=["tp"],
                shared=["cp"],
                reduce=["cp"],
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, local_sequence, dim))
        set_comm_metadata(
            self.cls_token,
            sharded=[None, None, None],
            shared=["tp", "cp"],
            reduce=["cp"],
        )
        set_comm_metadata(
            self.pos_embed,
            sharded=[None, "cp", None],
            shared=["tp"],
            reduce=[],
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                Block(dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        _annotate_replicated_norm(self.norm)
        self.head = nn.Linear(dim, num_classes)
        set_comm_metadata(
            self.head.weight,
            sharded=[None, None],
            shared=["tp", "cp"],
            reduce=["cp"],
        )
        set_comm_metadata(
            self.head.bias,
            sharded=[None],
            shared=["tp", "cp"],
            reduce=["cp"],
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _embed_local_patches(self, x: torch.Tensor) -> torch.Tensor:
        x = split_forward_allgather_backward(
            x, 2, self.input_height_shapes, "cp"
        )
        x = identity_forward_allreduce_backward(x, "tp")
        x = self.patch_embed(x)
        channel_shapes = [x.shape[1]] * comm.get_size("tp")
        x = allgather_forward_split_backward(
            x, 1, channel_shapes, "tp"
        )
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (
            x.shape[0],
            self.in_channels,
            self.image_size[0],
            self.image_size[1],
        )
        if tuple(x.shape) != expected:
            raise ValueError(f"expected input shape {expected}, got {tuple(x.shape)}")

        x = self._embed_local_patches(x)
        if comm.get_rank("cp") == 0:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        else:
            # Keep a zero gradient edge so every CP replica participates in the
            # declared CLS gradient reduction hook.
            x = x + self.cls_token.sum() * 0.0
        if x.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError("local token and positional shard lengths differ")
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x, self.sequence_shapes)
        x = self.norm(x)

        if comm.get_rank("cp") == 0:
            cls_hidden = x[:, 0]
        else:
            cls_hidden = x[:, 0] * 0.0
        cls_hidden = broadcast_forward_reduce_to_root_backward(
            cls_hidden, "cp"
        )
        return self.head(cls_hidden)

    @staticmethod
    def scale_loss_for_context_parallel(loss: torch.Tensor) -> torch.Tensor:
        """Scale duplicate CP losses before metadata-driven head reduction."""
        return loss / comm.get_size("cp")
