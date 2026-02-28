from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension."""

    def __init__(self, features: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"RMSNorm expected last dim {self.weight.shape[0]}, got {x.shape[-1]}."
            )
        rms = x.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight


class ChannelRMSNorm(nn.Module):
    """RMSNorm over channels for NCHW tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self._norm = RMSNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ChannelRMSNorm expects rank-4 NCHW tensor, got {x.shape}.")
        y = x.permute(0, 2, 3, 1)
        y = self._norm(y)
        return y.permute(0, 3, 1, 2)


class MLPBlock(nn.Module):
    """Linear + RMSNorm + SiLU."""

    def __init__(self, in_features: int, out_features: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features and out_features must be positive, got {in_features}, {out_features}."
            )
        self.linear = nn.Linear(in_features, out_features)
        self.norm = RMSNorm(out_features, eps=eps)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)))


class MLP(nn.Module):
    """Stack of MLP blocks with optional linear output head."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        out_features: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        blocks: list[nn.Module] = []
        current = in_features
        for _ in range(num_layers):
            blocks.append(MLPBlock(current, hidden_features, eps=eps))
            current = hidden_features
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(current, out_features) if out_features is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if self.head is not None:
            y = self.head(y)
        return y


class Conv2dBlock(nn.Module):
    """Stride-2 convolutional block used by Dreamer encoders."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got {in_channels}, {out_channels}."
            )
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = ChannelRMSNorm(out_channels, eps=eps)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvTranspose2dBlock(nn.Module):
    """Stride-2 transpose convolutional block used by Dreamer decoders."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got {in_channels}, {out_channels}."
            )
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = ChannelRMSNorm(out_channels, eps=eps)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.deconv(x)))


class BlockLinear(nn.Module):
    """Linear map with block-diagonal connectivity over the last feature dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        blocks: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features and out_features must be positive, got {in_features}, {out_features}."
            )
        if blocks <= 0:
            raise ValueError(f"blocks must be positive, got {blocks}.")
        if in_features % blocks != 0:
            raise ValueError(f"in_features={in_features} must be divisible by blocks={blocks}.")
        if out_features % blocks != 0:
            raise ValueError(f"out_features={out_features} must be divisible by blocks={blocks}.")

        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.in_per_block = in_features // blocks
        self.out_per_block = out_features // blocks

        self.weight = nn.Parameter(
            torch.empty(blocks, self.out_per_block, self.in_per_block),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for index in range(self.blocks):
            nn.init.xavier_uniform_(self.weight[index])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"BlockLinear expected last dim {self.in_features}, got {x.shape[-1]}."
            )
        base_shape = x.shape[:-1]
        x_blocks = x.reshape(*base_shape, self.blocks, self.in_per_block)
        y_blocks = torch.einsum("...bi,boi->...bo", x_blocks, self.weight)
        y = y_blocks.reshape(*base_shape, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y


class BlockGRUCell(nn.Module):
    """GRU cell with block-diagonal recurrent projections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        blocks: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError(
                f"input_size and hidden_size must be positive, got {input_size}, {hidden_size}."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.in_proj = nn.Linear(input_size, hidden_size * 3)
        self.h_proj = BlockLinear(hidden_size, hidden_size * 3, blocks=blocks, bias=False)
        self.in_norm = RMSNorm(hidden_size * 3, eps=eps)
        self.h_norm = RMSNorm(hidden_size * 3, eps=eps)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            raise ValueError(f"x last dim must be {self.input_size}, got {x.shape[-1]}.")
        if h.shape[-1] != self.hidden_size:
            raise ValueError(f"h last dim must be {self.hidden_size}, got {h.shape[-1]}.")
        if x.shape[:-1] != h.shape[:-1]:
            raise ValueError(f"x and h batch shapes must match, got {x.shape} and {h.shape}.")

        x_proj = self.in_norm(self.in_proj(x))
        h_proj = self.h_norm(self.h_proj(h))
        x_r, x_z, x_n = x_proj.chunk(3, dim=-1)
        h_r, h_z, h_n = h_proj.chunk(3, dim=-1)

        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        cand = torch.tanh(x_n + (reset * h_n))
        return (update * h) + ((1.0 - update) * cand)


def infer_image_shape(pixels: torch.Tensor) -> tuple[int, int, int]:
    """
    Infer channels/height/width from a rank-4 image tensor.
    Accepts NCHW or NHWC.
    """

    if pixels.ndim != 4:
        raise ValueError(f"Expected rank-4 image tensor, got {pixels.shape}.")
    # Heuristic: if last dim looks like channels, assume NHWC.
    if pixels.shape[-1] in {1, 3, 4}:
        return int(pixels.shape[-1]), int(pixels.shape[1]), int(pixels.shape[2])
    return int(pixels.shape[1]), int(pixels.shape[2]), int(pixels.shape[3])


def to_nchw(pixels: torch.Tensor) -> torch.Tensor:
    """Convert rank-4 image tensor to NCHW if needed."""

    if pixels.ndim != 4:
        raise ValueError(f"Expected rank-4 image tensor, got {pixels.shape}.")
    if pixels.shape[-1] in {1, 3, 4}:
        return pixels.permute(0, 3, 1, 2)
    return pixels


def flatten_last_dims(x: torch.Tensor, start_dim: int) -> torch.Tensor:
    if start_dim < 0 or start_dim >= x.ndim:
        raise ValueError(f"start_dim={start_dim} out of range for shape {x.shape}.")
    return x.flatten(start_dim=start_dim)


def ensure_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    if not hidden_dims:
        raise ValueError("hidden_dims must be non-empty.")
    dims = tuple(int(dim) for dim in hidden_dims)
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"hidden_dims must contain only positive ints, got {dims}.")
    return dims
