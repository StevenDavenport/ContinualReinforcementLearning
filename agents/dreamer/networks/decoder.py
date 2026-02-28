from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import BlockLinear, MLP, ConvTranspose2dBlock, RMSNorm, ensure_hidden_dims
from .distributions import symlog


@dataclass(frozen=True)
class DecoderConfig:
    """
    Dreamer-style observation decoder.

    - Image path: optional bspace projection -> transposed-conv stack -> sigmoid output.
    - Vector path: one MLP head per configured vector key.
    """

    feat_dim: int = 1536
    deter_dim: int = 512
    stoch_dim: int = 32
    classes: int = 32

    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    image_start_size: int = 4
    image_hidden_dims: tuple[int, ...] = (512, 256, 128, 64)
    bspace: int = 8

    vector_output_dim: int = 256
    vector_hidden_dim: int = 512
    vector_layers: int = 3
    vector_key: str = "proprio"
    # Optional multi-key vector outputs, e.g. (("proprio", 256), ("task_vec", 32)).
    vector_outputs: tuple[tuple[str, int], ...] = ()

    norm_eps: float = 1e-6


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if config.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {config.feat_dim}.")
        if config.deter_dim <= 0:
            raise ValueError(f"deter_dim must be positive, got {config.deter_dim}.")
        if config.stoch_dim <= 0 or config.classes <= 1:
            raise ValueError(
                f"Invalid stoch/classes values: stoch_dim={config.stoch_dim}, classes={config.classes}."
            )
        if config.feat_dim != config.deter_dim + (config.stoch_dim * config.classes):
            raise ValueError(
                "feat_dim must match deter_dim + stoch_dim * classes, got "
                f"feat_dim={config.feat_dim}, deter={config.deter_dim}, "
                f"stoch={config.stoch_dim}, classes={config.classes}."
            )
        if config.image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {config.image_channels}.")
        if config.image_height <= 0 or config.image_width <= 0:
            raise ValueError(
                "image_height and image_width must be positive, got "
                f"{config.image_height} and {config.image_width}."
            )
        if config.image_start_size <= 0:
            raise ValueError(
                f"image_start_size must be positive, got {config.image_start_size}."
            )
        if config.vector_output_dim < 0:
            raise ValueError(
                f"vector_output_dim must be >= 0, got {config.vector_output_dim}."
            )
        if config.vector_hidden_dim <= 0:
            raise ValueError(
                f"vector_hidden_dim must be positive, got {config.vector_hidden_dim}."
            )
        if config.vector_layers <= 0:
            raise ValueError(f"vector_layers must be positive, got {config.vector_layers}.")
        if config.bspace < 0:
            raise ValueError(f"bspace must be >= 0, got {config.bspace}.")
        if config.bspace > 0:
            if config.deter_dim % config.bspace != 0:
                raise ValueError(
                    f"deter_dim={config.deter_dim} must be divisible by bspace={config.bspace}."
                )

        self.config = config
        self.feat_dim = config.feat_dim
        self.deter_dim = config.deter_dim
        self.stoch_flat_dim = config.stoch_dim * config.classes

        hidden_dims = ensure_hidden_dims(config.image_hidden_dims)
        self._image_hidden_dims = hidden_dims
        self._start = config.image_start_size
        first_channels = hidden_dims[0]
        self._image_units = first_channels * self._start * self._start

        # Fallback dense projection path (used when bspace == 0).
        self.image_in = nn.Sequential(
            nn.Linear(config.feat_dim, self._image_units),
            RMSNorm(self._image_units, eps=config.norm_eps),
            nn.SiLU(),
        )

        # Dreamer bspace path: separate deter/stoch projections into spatial tensor.
        if config.bspace > 0:
            self.image_bspace_deter = BlockLinear(
                config.deter_dim,
                self._image_units,
                blocks=config.bspace,
            )
            self.image_bspace_stoch_1 = nn.Linear(self.stoch_flat_dim, 2 * config.vector_hidden_dim)
            self.image_bspace_stoch_1_norm = RMSNorm(2 * config.vector_hidden_dim, eps=config.norm_eps)
            self.image_bspace_stoch_2 = nn.Linear(2 * config.vector_hidden_dim, self._image_units)
            self.image_bspace_norm = RMSNorm(self._image_units, eps=config.norm_eps)
        else:
            self.image_bspace_deter = None
            self.image_bspace_stoch_1 = None
            self.image_bspace_stoch_1_norm = None
            self.image_bspace_stoch_2 = None
            self.image_bspace_norm = None

        deconv_layers: list[nn.Module] = []
        for in_ch, out_ch in zip(hidden_dims[:-1], hidden_dims[1:], strict=True):
            deconv_layers.append(ConvTranspose2dBlock(in_ch, out_ch, eps=config.norm_eps))
        self.image_backbone = nn.Sequential(*deconv_layers)
        self.image_out = nn.ConvTranspose2d(
            in_channels=hidden_dims[-1],
            out_channels=config.image_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        resolved_vector_outputs = self._resolve_vector_outputs(config)
        self.vector_decoders = nn.ModuleDict()
        self._vector_dims: dict[str, int] = {}
        for key, dim in resolved_vector_outputs.items():
            if dim <= 0:
                continue
            self.vector_decoders[key] = MLP(
                in_features=config.feat_dim,
                hidden_features=config.vector_hidden_dim,
                num_layers=config.vector_layers,
                out_features=dim,
                eps=config.norm_eps,
            )
            self._vector_dims[key] = dim

    def _resolve_vector_outputs(self, config: DecoderConfig) -> dict[str, int]:
        if config.vector_outputs:
            out: dict[str, int] = {}
            for key, dim in config.vector_outputs:
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Invalid vector output key: {key!r}.")
                if dim < 0:
                    raise ValueError(f"Vector output dim must be >= 0 for key={key!r}, got {dim}.")
                out[key] = int(dim)
            return out
        return {config.vector_key: int(config.vector_output_dim)}

    def _flatten_features(self, feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if feat.ndim < 2:
            raise ValueError(f"Expected rank >= 2 feature tensor (..., F), got {feat.shape}.")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Expected feat dim {self.feat_dim}, got {feat.shape[-1]}.")
        batch_shape = tuple(int(dim) for dim in feat.shape[:-1])
        return feat.reshape(-1, feat.shape[-1]), batch_shape

    @staticmethod
    def _restore_batch(x: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(*batch_shape, *x.shape[1:])

    def _image_start_tensor(self, flat: torch.Tensor) -> torch.Tensor:
        if self.image_bspace_deter is None:
            return self.image_in(flat)

        deter = flat[:, : self.deter_dim]
        stoch = flat[:, self.deter_dim :]

        x0 = self.image_bspace_deter(deter)

        assert self.image_bspace_stoch_1 is not None
        assert self.image_bspace_stoch_1_norm is not None
        assert self.image_bspace_stoch_2 is not None
        assert self.image_bspace_norm is not None
        x1 = self.image_bspace_stoch_1(stoch)
        x1 = F.silu(self.image_bspace_stoch_1_norm(x1))
        x1 = self.image_bspace_stoch_2(x1)

        return F.silu(self.image_bspace_norm(x0 + x1))

    def decode_pixels(self, feat: torch.Tensor) -> torch.Tensor:
        flat, batch_shape = self._flatten_features(feat)
        channels = self._image_hidden_dims[0]
        x = self._image_start_tensor(flat).reshape(flat.shape[0], channels, self._start, self._start)
        x = self.image_backbone(x)
        x = self.image_out(x)
        if x.shape[-2:] != (self.config.image_height, self.config.image_width):
            x = F.interpolate(
                x,
                size=(self.config.image_height, self.config.image_width),
                mode="bilinear",
                align_corners=False,
            )
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return self._restore_batch(x, batch_shape)

    def decode_vector(self, key: str, feat: torch.Tensor) -> torch.Tensor:
        if key not in self.vector_decoders:
            raise ValueError(f"Vector decoder for key={key!r} is disabled.")
        flat, batch_shape = self._flatten_features(feat)
        out = self.vector_decoders[key](flat)
        return self._restore_batch(out, batch_shape)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        outputs["pixels"] = self.decode_pixels(feat)
        for key in self.vector_decoders.keys():
            outputs[key] = self.decode_vector(key, feat)
        return outputs

    def align_vector_target(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self._vector_dims:
            raise ValueError(f"Unknown vector target key {key!r}.")
        if value.ndim < 2:
            raise ValueError(f"Expected rank >= 2 vector target (..., D), got {value.shape}.")
        target_dim = self._vector_dims[key]
        current_dim = int(value.shape[-1])
        if current_dim == target_dim:
            return value
        if current_dim > target_dim:
            return value[..., :target_dim]
        pad_shape = (*value.shape[:-1], target_dim - current_dim)
        pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        return torch.cat([value, pad], dim=-1)

    def preprocess_targets(
        self,
        observation: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        pixels = observation.get("pixels")
        if pixels is not None:
            pixels_value = pixels.to(dtype=torch.float32)
            if pixels_value.max().item() > 1.0:
                pixels_value = pixels_value / 255.0
            out["pixels"] = pixels_value

        for key in self.vector_decoders.keys():
            vector = observation.get(key)
            if vector is None:
                continue
            vec = symlog(vector.to(dtype=torch.float32))
            out[key] = self.align_vector_target(key, vec)
        return out


def infer_feat_dim(*, deter_dim: int, stoch_dim: int, classes: int) -> int:
    if deter_dim <= 0 or stoch_dim <= 0 or classes <= 1:
        raise ValueError(
            "Invalid RSSM dimensions for feature size inference: "
            f"deter_dim={deter_dim}, stoch_dim={stoch_dim}, classes={classes}."
        )
    return deter_dim + (stoch_dim * classes)


def infer_image_decoder_hidden_dims(
    *,
    model_dim: int,
    num_layers: int = 4,
) -> tuple[int, ...]:
    if model_dim <= 0:
        raise ValueError(f"model_dim must be positive, got {model_dim}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    dims: list[int] = [model_dim]
    for _ in range(num_layers - 1):
        dims.append(max(16, dims[-1] // 2))
    return tuple(dims)


def infer_vector_output_dim(
    observation: Mapping[str, torch.Tensor],
    *,
    key: str = "proprio",
) -> int:
    value = observation.get(key)
    if value is None:
        return 0
    if value.ndim < 1:
        raise ValueError(f"Expected vector rank >= 1 for key='{key}', got shape {value.shape}.")
    return int(value.shape[-1])


def merge_decoder_outputs(outputs: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    merged: dict[str, list[torch.Tensor]] = {}
    for output in outputs:
        for key, value in output.items():
            merged.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in merged.items()}
