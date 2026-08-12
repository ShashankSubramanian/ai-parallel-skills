"""Differentiable distributed primitives for the ViT."""

from .mappings import register_model_parallel_gradient_hooks

__all__ = ["register_model_parallel_gradient_hooks"]
