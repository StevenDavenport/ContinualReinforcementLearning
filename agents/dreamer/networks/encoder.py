from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .blocks import Conv2dBlock, MLPBlock, RMSNorm, ensure_hidden_dims, to_nchw
from .distributions import symlog


@dataclass(frozen=True)
class EncoderConfig:
    """
    DreamerV3-style encoder configuration.

    - Images: stride-2 conv stack.
    - Vectors: symlog + MLP.
    - If both are present, embeddings are fused by an MLP.
    """

    embed_dim: int = 1024
    image_channels: int = 3
    image_hidden_dims: tuple[int, ...] = (64, 128, 256, 512)
    image_pool_size: int = 4
    vector_input_dim: int = 256
    vector_hidden_dim: int = 512
    vector_layers: int = 3
    fusion_hidden_dim: int = 1024
    symlog_vector_input: bool = True
    norm_eps: float = 1e-6


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if config.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {config.embed_dim}.")
        if config.image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {config.image_channels}.")
        if config.vector_hidden_dim <= 0:
            raise ValueError(f"vector_hidden_dim must be positive, got {config.vector_hidden_dim}.")
        if config.vector_input_dim <= 0:
            raise ValueError(f"vector_input_dim must be positive, got {config.vector_input_dim}.")
        if config.image_pool_size <= 0:
            raise ValueError(f"image_pool_size must be positive, got {config.image_pool_size}.")
        if config.vector_layers <= 0:
            raise ValueError(f"vector_layers must be positive, got {config.vector_layers}.")
        if config.fusion_hidden_dim <= 0:
            raise ValueError(f"fusion_hidden_dim must be positive, got {config.fusion_hidden_dim}.")

        self.config = config
        self.embed_dim = config.embed_dim

        image_hidden_dims = ensure_hidden_dims(config.image_hidden_dims)
        image_layers: list[nn.Module] = []
        in_ch = config.image_channels
        for out_ch in image_hidden_dims:
            image_layers.append(Conv2dBlock(in_ch, out_ch, eps=config.norm_eps))
            in_ch = out_ch
        pooled_hw = config.image_pool_size
        image_features = image_hidden_dims[-1] * pooled_hw * pooled_hw
        self.image_backbone = nn.Sequential(*image_layers)
        self.image_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((pooled_hw, pooled_hw)),
            nn.Flatten(start_dim=1),
            nn.Linear(image_features, config.embed_dim),
            RMSNorm(config.embed_dim, eps=config.norm_eps),
            nn.SiLU(),
        )

        vector_layers: list[nn.Module] = [
            nn.Linear(config.vector_input_dim, config.vector_hidden_dim),
            RMSNorm(config.vector_hidden_dim, eps=config.norm_eps),
            nn.SiLU(),
        ]
        for _ in range(config.vector_layers - 1):
            vector_layers.append(
                MLPBlock(
                    config.vector_hidden_dim,
                    config.vector_hidden_dim,
                    eps=config.norm_eps,
                )
            )
        vector_layers.extend(
            [
                nn.Linear(config.vector_hidden_dim, config.embed_dim),
                RMSNorm(config.embed_dim, eps=config.norm_eps),
                nn.SiLU(),
            ]
        )
        self.vector_encoder = nn.Sequential(*vector_layers)

        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.fusion_hidden_dim),
            RMSNorm(config.fusion_hidden_dim, eps=config.norm_eps),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.embed_dim),
        )

    def _module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        reference = next(self.parameters())
        return reference.device, reference.dtype

    def _to_float_tensor(self, value: Any) -> torch.Tensor:
        device, dtype = self._module_device_dtype()
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _flatten_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim < 2:
            raise ValueError(f"Expected rank >= 2 tensor, got {x.shape}.")
        batch_shape = tuple(int(dim) for dim in x.shape[:-1])
        flat = x.reshape(-1, x.shape[-1])
        return flat, batch_shape

    def _restore_batch(self, x: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(*batch_shape, x.shape[-1])

    def encode_vector(self, vector: Any) -> torch.Tensor:
        vector_tensor = self._to_float_tensor(vector)
        if vector_tensor.ndim < 2:
            raise ValueError(
                f"Vector inputs must have rank >= 2 (..., D). Got {vector_tensor.shape}."
            )
        if self.config.symlog_vector_input:
            vector_tensor = symlog(vector_tensor)
        vector_tensor = self._fixed_vector_dim(vector_tensor)

        flat, batch_shape = self._flatten_batch(vector_tensor)
        encoded = self.vector_encoder(flat)
        return self._restore_batch(encoded, batch_shape)

    def _fixed_vector_dim(self, vector_tensor: torch.Tensor) -> torch.Tensor:
        feature_dim = int(vector_tensor.shape[-1])
        target_dim = self.config.vector_input_dim
        if feature_dim == target_dim:
            return vector_tensor
        if feature_dim > target_dim:
            return vector_tensor[..., :target_dim]
        pad_shape = (*vector_tensor.shape[:-1], target_dim - feature_dim)
        pad = torch.zeros(
            pad_shape,
            dtype=vector_tensor.dtype,
            device=vector_tensor.device,
        )
        return torch.cat([vector_tensor, pad], dim=-1)

    def _flatten_image_batch(self, pixels: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if pixels.ndim < 4:
            raise ValueError(
                f"Pixels must have rank >= 4 (..., H, W, C) or (..., C, H, W). Got {pixels.shape}."
            )

        if pixels.shape[-1] in {1, 3, 4} or pixels.shape[-3] in {1, 3, 4}:
            batch_shape = tuple(int(dim) for dim in pixels.shape[:-3])
            flat = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
        else:
            raise ValueError(
                "Could not infer channel position in pixels tensor. "
                f"Expected channel dim to be size in {{1,3,4}}, got shape {pixels.shape}."
            )
        return flat, batch_shape

    def encode_pixels(self, pixels: Any) -> torch.Tensor:
        pixels_tensor = self._to_float_tensor(pixels)
        flat, batch_shape = self._flatten_image_batch(pixels_tensor)
        flat = to_nchw(flat)
        if flat.shape[1] != self.config.image_channels:
            raise ValueError(
                f"Expected {self.config.image_channels} image channels, got {flat.shape[1]}."
            )

        if flat.max().item() > 1.0:
            flat = (flat / 255.0) - 0.5
        else:
            flat = flat - 0.5
        features = self.image_backbone(flat)
        encoded = self.image_proj(features)
        return self._restore_batch(encoded, batch_shape)

    def forward(self, observation: Mapping[str, Any]) -> torch.Tensor:
        pixels = observation.get("pixels")
        vector = observation.get("proprio")

        image_embed: torch.Tensor | None = None
        vector_embed: torch.Tensor | None = None

        if pixels is not None:
            image_embed = self.encode_pixels(pixels)
        if vector is not None:
            vector_embed = self.encode_vector(vector)

        if image_embed is None and vector_embed is None:
            raise ValueError(
                "Encoder expects at least one of observation['pixels'] or observation['proprio']."
            )
        if image_embed is None:
            assert vector_embed is not None
            return vector_embed
        if vector_embed is None:
            return image_embed
        if image_embed.shape[:-1] != vector_embed.shape[:-1]:
            raise ValueError(
                "Pixels/proprio batch dimensions must match, got "
                f"{image_embed.shape[:-1]} and {vector_embed.shape[:-1]}."
            )
        return self.fusion(torch.cat([image_embed, vector_embed], dim=-1))
