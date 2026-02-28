from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class LaPropConfig:
    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: bool = True
    nesterov: bool = False
    wd: float = 0.0
    wdregex: str = r"/kernel$"
    schedule: str = "const"
    warmup: int = 1000
    anneal: int = 0
    pmin: float = 1e-3


class LaProp(torch.optim.Optimizer):
    """
    DreamerV3-style optimizer chain:
    AGC -> RMS scaling -> momentum -> optional decay -> LR schedule.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: LaPropConfig,
    ) -> None:
        params: list[torch.nn.Parameter] = []
        self._param_names: dict[int, str] = {}
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            params.append(param)
            self._param_names[id(param)] = name
        if not params:
            raise ValueError("LaProp received no trainable parameters.")

        if config.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {config.lr}.")
        if config.agc < 0.0:
            raise ValueError(f"agc must be >= 0, got {config.agc}.")
        if config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {config.eps}.")
        if not (0.0 <= config.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {config.beta1}.")
        if not (0.0 <= config.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {config.beta2}.")
        if config.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {config.warmup}.")
        if config.anneal < 0:
            raise ValueError(f"anneal must be >= 0, got {config.anneal}.")
        if config.schedule not in {"const", "linear", "cosine"}:
            raise ValueError(f"Unsupported schedule: {config.schedule!r}.")
        if config.pmin <= 0.0:
            raise ValueError(f"pmin must be positive, got {config.pmin}.")

        super().__init__(params, defaults={})
        self.config = config
        self._updates = 0
        self._wd_pattern = re.compile(config.wdregex) if config.wd > 0.0 else None
        self._last_lr = 0.0

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def last_lr(self) -> float:
        return self._last_lr

    def _scheduled_lr(self, step: int) -> float:
        lr = self.config.lr

        if self.config.schedule == "const":
            base = lr
        else:
            if self.config.anneal <= self.config.warmup:
                base = lr
            else:
                span = max(1, self.config.anneal - self.config.warmup)
                x = min(max((step - self.config.warmup) / float(span), 0.0), 1.0)
                if self.config.schedule == "linear":
                    base = ((1.0 - x) * lr) + (x * (0.1 * lr))
                else:
                    cosine = 0.5 * (1.0 + math.cos(math.pi * x))
                    base = (0.1 * lr) + (cosine * (lr - (0.1 * lr)))

        if self.config.warmup > 0 and step <= self.config.warmup:
            ramp = lr * (step / float(self.config.warmup))
            return ramp
        return base

    def _decay_applies(self, param: torch.nn.Parameter) -> bool:
        if self._wd_pattern is None:
            return False
        name = self._param_names.get(id(param), "")
        return bool(self._wd_pattern.search(name))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._updates += 1
        lr = self._scheduled_lr(self._updates)
        self._last_lr = lr

        beta1 = self.config.beta1
        beta2 = self.config.beta2
        eps = self.config.eps

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients.")

                if self.config.agc > 0.0:
                    grad_norm = torch.linalg.vector_norm(grad)
                    param_norm = torch.linalg.vector_norm(param.detach())
                    upper = self.config.agc * torch.maximum(
                        torch.tensor(self.config.pmin, device=param.device, dtype=param.dtype),
                        param_norm,
                    )
                    scale = 1.0 / torch.maximum(
                        torch.ones((), device=param.device, dtype=param.dtype),
                        grad_norm / upper,
                    )
                    grad = grad * scale

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["nu"] = torch.zeros_like(param)
                    state["mu"] = torch.zeros_like(param)

                state["step"] += 1
                step = state["step"]
                nu: torch.Tensor = state["nu"]
                mu: torch.Tensor = state["mu"]

                nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                nu_hat = nu / (1.0 - (beta2**step))
                update = grad / (nu_hat.sqrt() + eps)

                if self.config.momentum:
                    mu.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    if self.config.nesterov:
                        mu_nesterov = (beta1 * mu) + ((1.0 - beta1) * update)
                        update = mu_nesterov / (1.0 - (beta1**step))
                    else:
                        update = mu / (1.0 - (beta1**step))

                if self.config.wd > 0.0 and self._decay_applies(param):
                    update = update + (self.config.wd * param.detach())

                param.add_(update, alpha=-lr)

        return loss

    def state_dict(self):  # type: ignore[override]
        data = super().state_dict()
        data["updates"] = self._updates
        data["last_lr"] = self._last_lr
        return data

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self._updates = int(state_dict.pop("updates", 0))
        self._last_lr = float(state_dict.pop("last_lr", 0.0))
        return super().load_state_dict(state_dict)
