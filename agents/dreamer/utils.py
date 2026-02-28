from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .networks.distributions import symexp as _symexp
from .networks.distributions import symlog as _symlog

_OBS_IGNORE_KEYS = frozenset(
    {
        "pixels",
        "proprio",
        "task",
        "domain_name",
        "env_family",
        "env_option",
        "step",
        "action_space_n",
        "action_dim",
        "action_low",
        "action_high",
        "continuous_action",
    }
)


def symlog(value: torch.Tensor) -> torch.Tensor:
    return _symlog(value)


def symexp(value: torch.Tensor) -> torch.Tensor:
    return _symexp(value)


def align_last_dim(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    pad_shape = (*value.shape[:-1], dim - value.shape[-1])
    pad = torch.zeros(pad_shape, device=value.device, dtype=value.dtype)
    return torch.cat([value, pad], dim=-1)


def _is_numeric_scalar(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _flatten_numeric(value: object, *, out: list[float], max_items: int) -> None:
    if len(out) >= max_items:
        return
    if _is_numeric_scalar(value):
        numeric = float(value)
        if math.isfinite(numeric):
            out.append(numeric)
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            if len(out) >= max_items:
                break
            _flatten_numeric(value[key], out=out, max_items=max_items)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            if len(out) >= max_items:
                break
            _flatten_numeric(item, out=out, max_items=max_items)


def _to_float_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _proprio_from_observation(
    observation: Mapping[str, Any],
    *,
    vector_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if "proprio" in observation:
        proprio = _to_float_tensor(observation["proprio"], device=device, dtype=dtype)
        proprio = proprio.reshape(1) if proprio.ndim == 0 else proprio.reshape(-1)
        return align_last_dim(proprio, dim=vector_dim)

    flattened: list[float] = []
    filtered = {
        key: value for key, value in observation.items() if key not in _OBS_IGNORE_KEYS
    }
    _flatten_numeric(filtered, out=flattened, max_items=vector_dim)
    if not flattened:
        flattened = [0.0]
    tensor = torch.tensor(flattened, device=device, dtype=dtype)
    return align_last_dim(tensor, dim=vector_dim)


def single_observation_to_model_input(
    observation: Mapping[str, Any],
    *,
    vector_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """
    Convert one raw environment observation into encoder-ready batched tensors.
    """

    if vector_dim <= 0:
        raise ValueError(f"vector_dim must be positive, got {vector_dim}.")

    out: dict[str, torch.Tensor] = {}
    pixels = observation.get("pixels")
    if pixels is not None:
        out["pixels"] = _to_float_tensor(pixels, device=device, dtype=dtype).unsqueeze(0)
    out["proprio"] = _proprio_from_observation(
        observation,
        vector_dim=vector_dim,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    return out


def batch_observations_to_model_input(  # noqa: PLR0912
    observations: Mapping[str, torch.Tensor],
    *,
    vector_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """
    Convert replay batch observations to encoder/decoder-compatible tensors.
    Expects leading [T, B] sequence dimensions.
    """

    if vector_dim <= 0:
        raise ValueError(f"vector_dim must be positive, got {vector_dim}.")

    out: dict[str, torch.Tensor] = {}

    pixels = observations.get("pixels")
    if pixels is not None:
        out["pixels"] = _to_float_tensor(pixels, device=device, dtype=dtype)

    if "proprio" in observations:
        proprio = _to_float_tensor(observations["proprio"], device=device, dtype=dtype)
        if proprio.ndim < 3:
            raise ValueError(
                "Replay proprio must have rank >= 3 [T,B,...], got "
                f"{tuple(proprio.shape)}."
            )
        proprio = proprio.reshape(*proprio.shape[:2], -1)
        out["proprio"] = align_last_dim(proprio, dim=vector_dim)
        return out

    components: list[torch.Tensor] = []
    base_shape: tuple[int, int] | None = None
    for key in sorted(observations):
        if key in _OBS_IGNORE_KEYS:
            continue
        value = observations[key]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = _to_float_tensor(value, device=device, dtype=dtype)
        if tensor.ndim < 2:
            continue
        if base_shape is None:
            base_shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        if base_shape != (int(tensor.shape[0]), int(tensor.shape[1])):
            continue
        components.append(tensor.reshape(*tensor.shape[:2], -1))

    if components:
        proprio = torch.cat(components, dim=-1)
        out["proprio"] = align_last_dim(proprio, dim=vector_dim)
        return out

    if base_shape is None:
        if "pixels" in out:
            pixel_shape = out["pixels"].shape
            if len(pixel_shape) < 2:
                raise ValueError("Unable to infer [T, B] from pixels replay tensor.")
            base_shape = (int(pixel_shape[0]), int(pixel_shape[1]))
        else:
            base_shape = (1, 1)
    out["proprio"] = torch.zeros(*base_shape, vector_dim, device=device, dtype=dtype)
    return out


def action_to_model_vector(  # noqa: PLR0912
    action: object,
    *,
    action_space: str,
    action_dim: int,
) -> list[float]:
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if action_space not in {"discrete", "continuous"}:
        raise ValueError(f"Unsupported action_space: {action_space!r}.")

    if action_space == "discrete":
        if isinstance(action, torch.Tensor):
            if action.numel() == action_dim:
                index = int(action.reshape(-1).argmax().item())
            elif action.numel() >= 1:
                index = int(action.reshape(-1)[0].item())
            else:
                index = 0
        elif isinstance(action, int | float) and not isinstance(action, bool):
            index = int(action)
        elif isinstance(action, Sequence) and not isinstance(action, str | bytes | bytearray):
            values = [float(item) for item in action]
            if not values:
                index = 0
            elif len(values) == action_dim:
                index = int(max(range(len(values)), key=lambda i: values[i]))
            else:
                index = int(values[0])
        else:
            index = 0
        index = max(0, min(action_dim - 1, index))
        onehot = [0.0 for _ in range(action_dim)]
        onehot[index] = 1.0
        return onehot

    values: list[float]
    if isinstance(action, torch.Tensor):
        values = [float(item) for item in action.reshape(-1).tolist()]
    elif isinstance(action, Sequence) and not isinstance(action, str | bytes | bytearray):
        values = [float(item) for item in action]
    elif isinstance(action, int | float) and not isinstance(action, bool):
        values = [float(action)]
    else:
        values = []
    if len(values) < action_dim:
        values.extend([0.0] * (action_dim - len(values)))
    values = values[:action_dim]
    return [max(-1.0, min(1.0, value)) for value in values]


def env_action_from_model_tensor(
    action: torch.Tensor,
    *,
    action_space: str,
) -> int | list[float]:
    if action_space not in {"discrete", "continuous"}:
        raise ValueError(f"Unsupported action_space: {action_space!r}.")
    if action_space == "discrete":
        if action.numel() == 0:
            return 0
        return int(action.reshape(-1)[0].item())
    if action.ndim == 0:
        return [float(action.item())]
    if action.ndim == 1:
        return [float(item) for item in action.tolist()]
    return [float(item) for item in action[0].tolist()]
