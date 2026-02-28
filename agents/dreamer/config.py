from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .losses import (
    ActorLossConfig,
    CriticLossConfig,
    NormalizeConfig,
    PredictionLossConfig,
    WorldModelLossConfig,
)
from .networks.actor import ActorConfig
from .networks.critic import CriticConfig
from .networks.decoder import DecoderConfig
from .networks.encoder import EncoderConfig
from .networks.rssm import RSSMConfig
from .optim import LaPropConfig


def _require_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be int, got {value!r}.")
    return value


def _require_float(name: str, value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    return float(value)


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool, got {value!r}.")
    return value


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be str, got {value!r}.")
    return value


def _parse_vector_outputs(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    items: list[tuple[str, int]] = []
    if isinstance(value, Mapping):
        iterable = sorted(value.items(), key=lambda item: str(item[0]))
    elif isinstance(value, tuple | list):
        iterable = value
    else:
        raise ValueError(
            "vector_outputs must be mapping or sequence of (key, dim), "
            f"got {type(value).__name__}."
        )

    for item in iterable:
        if isinstance(item, tuple | list) and len(item) == 2:
            key_raw, dim_raw = item
        elif isinstance(value, Mapping):
            key_raw, dim_raw = item
        else:
            raise ValueError(f"Invalid vector_outputs entry: {item!r}.")
        key = _require_str("vector_outputs key", key_raw).strip()
        dim = _require_int(f"vector_outputs[{key}]", dim_raw)
        if dim < 0:
            raise ValueError(f"vector_outputs[{key}] must be >= 0, got {dim}.")
        items.append((key, dim))
    return tuple(items)


@dataclass(frozen=True)
class DreamerConfig:
    seed: int = 0
    device: str = "cpu"
    compute_dtype: str = "float32"

    action_space: str = "discrete"
    action_dim: int = 32

    model_dim: int = 512
    embed_dim: int = 1024
    deter_dim: int = 512
    stoch_dim: int = 32
    classes: int = 32
    blocks: int = 8
    unimix: float = 0.01

    image_channels: int = 3
    image_size: int = 64
    vector_input_dim: int = 256
    vector_output_dim: int = 256
    vector_outputs: tuple[tuple[str, int], ...] = ()

    batch_size: int = 16
    batch_length: int = 64
    replay_capacity: int = 5_000_000
    replay_online: bool = True
    replay_uniform_frac: float = 1.0
    replay_recency_frac: float = 0.0
    replay_recency_exp: float = 1.0
    replay_chunksize: int = 1024
    replay_context: int = 1
    warmup_steps: int = 256
    train_ratio: float = 32.0
    imagine_horizon: int = 15
    imag_last: int = 0

    horizon: int = 333
    contdisc: bool = True
    ac_grads: bool = False
    reward_grad: bool = True
    repval_loss: bool = True
    repval_grad: bool = True
    report: bool = True
    report_gradnorms: bool = False

    model_lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5
    opt_agc: float = 0.3
    opt_eps: float = 1e-20
    opt_beta1: float = 0.9
    opt_beta2: float = 0.999
    opt_momentum: bool = True
    opt_nesterov: bool = False
    opt_wd: float = 0.0
    opt_wdregex: str = r"/kernel$"
    opt_schedule: str = "const"
    opt_warmup: int = 1000
    opt_anneal: int = 0

    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    actor_outscale: float = 0.01
    rssm_outscale: float = 1.0
    reward_outscale: float = 0.0
    continue_outscale: float = 1.0
    value_outscale: float = 0.0

    reward_bins: int = 255
    reward_log_low: float = -20.0
    reward_log_high: float = 20.0

    beta_pred: float = 1.0
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    free_nats: float = 1.0

    actent: float = 3e-4
    lambda_: float = 0.95
    slowreg: float = 1.0
    slowtar: bool = False
    repval_scale: float = 0.3

    slowvalue_rate: float = 0.02
    slowvalue_every: int = 1

    retnorm_impl: str = "perc"
    retnorm_rate: float = 0.01
    retnorm_limit: float = 1.0
    retnorm_perclo: float = 5.0
    retnorm_perchi: float = 95.0
    retnorm_debias: bool = False

    valnorm_impl: str = "none"
    valnorm_rate: float = 0.01
    valnorm_limit: float = 1e-8
    valnorm_perclo: float = 5.0
    valnorm_perchi: float = 95.0
    valnorm_debias: bool = True

    advnorm_impl: str = "none"
    advnorm_rate: float = 0.01
    advnorm_limit: float = 1e-8
    advnorm_perclo: float = 5.0
    advnorm_perchi: float = 95.0
    advnorm_debias: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device string: {self.device!r}.")
        if self.compute_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                f"compute_dtype must be one of float32/bfloat16/float16, got {self.compute_dtype!r}."
            )
        if self.action_space not in {"discrete", "continuous"}:
            raise ValueError(
                f"action_space must be 'discrete' or 'continuous', got {self.action_space!r}."
            )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}.")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}.")
        if self.deter_dim <= 0:
            raise ValueError(f"deter_dim must be positive, got {self.deter_dim}.")
        if self.stoch_dim <= 0:
            raise ValueError(f"stoch_dim must be positive, got {self.stoch_dim}.")
        if self.classes <= 1:
            raise ValueError(f"classes must be > 1, got {self.classes}.")
        if self.blocks <= 0:
            raise ValueError(f"blocks must be positive, got {self.blocks}.")
        if not (0.0 <= self.unimix < 1.0):
            raise ValueError(f"unimix must be in [0, 1), got {self.unimix}.")
        if self.image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {self.image_channels}.")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}.")
        if self.vector_input_dim <= 0:
            raise ValueError(f"vector_input_dim must be positive, got {self.vector_input_dim}.")
        if self.vector_output_dim < 0:
            raise ValueError(f"vector_output_dim must be >= 0, got {self.vector_output_dim}.")
        for key, dim in self.vector_outputs:
            if not key:
                raise ValueError("vector_outputs keys must be non-empty.")
            if dim < 0:
                raise ValueError(f"vector_outputs[{key}] must be >= 0, got {dim}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.batch_length <= 1:
            raise ValueError(f"batch_length must be > 1, got {self.batch_length}.")
        if self.replay_capacity <= 0:
            raise ValueError(f"replay_capacity must be positive, got {self.replay_capacity}.")
        if self.replay_uniform_frac < 0.0 or self.replay_recency_frac < 0.0:
            raise ValueError(
                "replay_uniform_frac and replay_recency_frac must be non-negative."
            )
        if self.replay_uniform_frac + self.replay_recency_frac <= 0.0:
            raise ValueError(
                "At least one replay fraction must be positive "
                "(replay_uniform_frac + replay_recency_frac > 0)."
            )
        if self.replay_recency_exp <= 0.0:
            raise ValueError(
                f"replay_recency_exp must be positive, got {self.replay_recency_exp}."
            )
        if self.replay_chunksize <= 0:
            raise ValueError(f"replay_chunksize must be positive, got {self.replay_chunksize}.")
        if self.replay_context < 0:
            raise ValueError(f"replay_context must be >= 0, got {self.replay_context}.")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")
        if self.train_ratio <= 0.0:
            raise ValueError(f"train_ratio must be positive, got {self.train_ratio}.")
        if self.imagine_horizon <= 0:
            raise ValueError(f"imagine_horizon must be positive, got {self.imagine_horizon}.")
        if self.imag_last < 0:
            raise ValueError(f"imag_last must be >= 0, got {self.imag_last}.")
        if self.horizon <= 1:
            raise ValueError(f"horizon must be > 1, got {self.horizon}.")
        if self.model_lr <= 0.0 or self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if self.actor_min_std <= 0.0:
            raise ValueError(f"actor_min_std must be > 0, got {self.actor_min_std}.")
        if self.actor_max_std <= self.actor_min_std:
            raise ValueError(
                f"actor_max_std must be > actor_min_std, got {self.actor_max_std} <= {self.actor_min_std}."
            )
        if self.actor_outscale < 0.0:
            raise ValueError(f"actor_outscale must be >= 0, got {self.actor_outscale}.")
        if self.rssm_outscale < 0.0:
            raise ValueError(f"rssm_outscale must be >= 0, got {self.rssm_outscale}.")
        if self.reward_outscale < 0.0:
            raise ValueError(f"reward_outscale must be >= 0, got {self.reward_outscale}.")
        if self.continue_outscale < 0.0:
            raise ValueError(f"continue_outscale must be >= 0, got {self.continue_outscale}.")
        if self.value_outscale < 0.0:
            raise ValueError(f"value_outscale must be >= 0, got {self.value_outscale}.")
        if self.reward_bins < 2:
            raise ValueError(f"reward_bins must be >= 2, got {self.reward_bins}.")
        if not (self.reward_log_low < self.reward_log_high):
            raise ValueError(
                f"reward_log_low must be < reward_log_high, got {self.reward_log_low} and {self.reward_log_high}."
            )
        if self.free_nats <= 0.0:
            raise ValueError(f"free_nats must be > 0, got {self.free_nats}.")
        if not (0.0 <= self.lambda_ <= 1.0):
            raise ValueError(f"lambda must be in [0,1], got {self.lambda_}.")

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + (self.stoch_dim * self.classes)

    @property
    def train_updates_per_env_step(self) -> float:
        return self.train_ratio / float(self.batch_length)

    @property
    def torch_dtype(self):
        import torch

        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.compute_dtype]

    def build_encoder_config(self) -> EncoderConfig:
        return EncoderConfig(
            embed_dim=self.embed_dim,
            image_channels=self.image_channels,
            vector_input_dim=self.vector_input_dim,
            vector_hidden_dim=self.model_dim,
            vector_layers=3,
            fusion_hidden_dim=self.embed_dim,
        )

    def build_rssm_config(self) -> RSSMConfig:
        return RSSMConfig(
            deter_dim=self.deter_dim,
            stoch_dim=self.stoch_dim,
            classes=self.classes,
            hidden_dim=self.model_dim,
            action_dim=self.action_dim,
            embed_dim=self.embed_dim,
            blocks=self.blocks,
            unimix=self.unimix,
            img_layers=2,
            obs_layers=1,
            dyn_layers=1,
            absolute=False,
            outscale=self.rssm_outscale,
        )

    def build_decoder_config(self) -> DecoderConfig:
        return DecoderConfig(
            feat_dim=self.feat_dim,
            deter_dim=self.deter_dim,
            stoch_dim=self.stoch_dim,
            classes=self.classes,
            image_channels=self.image_channels,
            image_height=self.image_size,
            image_width=self.image_size,
            image_hidden_dims=(self.model_dim, self.model_dim // 2, self.model_dim // 4, 64),
            vector_output_dim=self.vector_output_dim,
            vector_hidden_dim=self.model_dim,
            vector_layers=3,
            vector_key="proprio",
            vector_outputs=self.vector_outputs,
            bspace=self.blocks,
        )

    def build_actor_config(self) -> ActorConfig:
        return ActorConfig(
            feat_dim=self.feat_dim,
            hidden_dim=self.model_dim,
            hidden_layers=3,
            action_dim=self.action_dim,
            action_space=self.action_space,
            unimix=self.unimix,
            min_std=self.actor_min_std,
            max_std=self.actor_max_std,
            outscale=self.actor_outscale,
        )

    def build_critic_config(self) -> CriticConfig:
        return CriticConfig(
            feat_dim=self.feat_dim,
            hidden_dim=self.model_dim,
            hidden_layers=3,
            bins=self.reward_bins,
            log_low=self.reward_log_low,
            log_high=self.reward_log_high,
            outscale=self.value_outscale,
        )

    def build_world_model_loss_config(self) -> WorldModelLossConfig:
        return WorldModelLossConfig(
            beta_pred=self.beta_pred,
            beta_dyn=self.beta_dyn,
            beta_rep=self.beta_rep,
            free_nats=self.free_nats,
            prediction=PredictionLossConfig(
                reward_bins=self.reward_bins,
                reward_log_low=self.reward_log_low,
                reward_log_high=self.reward_log_high,
            ),
        )

    def build_actor_loss_config(self) -> ActorLossConfig:
        return ActorLossConfig(
            contdisc=self.contdisc,
            horizon=self.horizon,
            lam=self.lambda_,
            actent=self.actent,
            slowreg=self.slowreg,
            slowtar=self.slowtar,
        )

    def build_critic_loss_config(self) -> CriticLossConfig:
        return CriticLossConfig(
            bins=self.reward_bins,
            log_low=self.reward_log_low,
            log_high=self.reward_log_high,
            lam=self.lambda_,
            slowreg=self.slowreg,
            slowtar=self.slowtar,
            horizon=self.horizon,
            repval_scale=self.repval_scale,
        )

    def build_retnorm_config(self) -> NormalizeConfig:
        return NormalizeConfig(
            impl=self.retnorm_impl,
            rate=self.retnorm_rate,
            limit=self.retnorm_limit,
            perclo=self.retnorm_perclo,
            perchi=self.retnorm_perchi,
            debias=self.retnorm_debias,
        )

    def build_valnorm_config(self) -> NormalizeConfig:
        return NormalizeConfig(
            impl=self.valnorm_impl,
            rate=self.valnorm_rate,
            limit=self.valnorm_limit,
            perclo=self.valnorm_perclo,
            perchi=self.valnorm_perchi,
            debias=self.valnorm_debias,
        )

    def build_advnorm_config(self) -> NormalizeConfig:
        return NormalizeConfig(
            impl=self.advnorm_impl,
            rate=self.advnorm_rate,
            limit=self.advnorm_limit,
            perclo=self.advnorm_perclo,
            perchi=self.advnorm_perchi,
            debias=self.advnorm_debias,
        )

    def build_model_opt_config(self) -> LaPropConfig:
        return LaPropConfig(
            lr=self.model_lr,
            agc=self.opt_agc,
            eps=self.opt_eps,
            beta1=self.opt_beta1,
            beta2=self.opt_beta2,
            momentum=self.opt_momentum,
            nesterov=self.opt_nesterov,
            wd=self.opt_wd,
            wdregex=self.opt_wdregex,
            schedule=self.opt_schedule,
            warmup=self.opt_warmup,
            anneal=self.opt_anneal,
        )

    def build_actor_opt_config(self) -> LaPropConfig:
        return LaPropConfig(
            lr=self.actor_lr,
            agc=self.opt_agc,
            eps=self.opt_eps,
            beta1=self.opt_beta1,
            beta2=self.opt_beta2,
            momentum=self.opt_momentum,
            nesterov=self.opt_nesterov,
            wd=self.opt_wd,
            wdregex=self.opt_wdregex,
            schedule=self.opt_schedule,
            warmup=self.opt_warmup,
            anneal=self.opt_anneal,
        )

    def build_critic_opt_config(self) -> LaPropConfig:
        return LaPropConfig(
            lr=self.critic_lr,
            agc=self.opt_agc,
            eps=self.opt_eps,
            beta1=self.opt_beta1,
            beta2=self.opt_beta2,
            momentum=self.opt_momentum,
            nesterov=self.opt_nesterov,
            wd=self.opt_wd,
            wdregex=self.opt_wdregex,
            schedule=self.opt_schedule,
            warmup=self.opt_warmup,
            anneal=self.opt_anneal,
        )


def parse_dreamer_config(raw: Mapping[str, object]) -> DreamerConfig:  # noqa: PLR0915
    def get_int(name: str, default: int) -> int:
        return _require_int(name, raw.get(name, default))

    def get_float(name: str, default: float) -> float:
        return _require_float(name, raw.get(name, default))

    def get_bool(name: str, default: bool) -> bool:
        return _require_bool(name, raw.get(name, default))

    def get_str(name: str, default: str) -> str:
        return _require_str(name, raw.get(name, default)).strip()

    return DreamerConfig(
        seed=get_int("seed", 0),
        device=get_str("device", "cpu"),
        compute_dtype=get_str("compute_dtype", "float32"),
        action_space=get_str("action_space", "discrete"),
        action_dim=get_int("action_dim", 32),
        model_dim=get_int("model_dim", 512),
        embed_dim=get_int("embed_dim", 1024),
        deter_dim=get_int("deter_dim", 512),
        stoch_dim=get_int("stoch_dim", 32),
        classes=get_int("classes", 32),
        blocks=get_int("blocks", 8),
        unimix=get_float("unimix", 0.01),
        image_channels=get_int("image_channels", 3),
        image_size=get_int("image_size", 64),
        vector_input_dim=get_int("vector_input_dim", 256),
        vector_output_dim=get_int("vector_output_dim", 256),
        vector_outputs=_parse_vector_outputs(raw.get("vector_outputs", ())),
        batch_size=get_int("batch_size", 16),
        batch_length=get_int("batch_length", 64),
        replay_capacity=get_int("replay_capacity", 5_000_000),
        replay_online=get_bool("replay_online", True),
        replay_uniform_frac=get_float("replay_uniform_frac", 1.0),
        replay_recency_frac=get_float("replay_recency_frac", 0.0),
        replay_recency_exp=get_float("replay_recency_exp", 1.0),
        replay_chunksize=get_int("replay_chunksize", 1024),
        replay_context=get_int("replay_context", 1),
        warmup_steps=get_int("warmup_steps", 256),
        train_ratio=get_float("train_ratio", 32.0),
        imagine_horizon=get_int("imagine_horizon", 15),
        imag_last=get_int("imag_last", 0),
        horizon=get_int("horizon", 333),
        contdisc=get_bool("contdisc", True),
        ac_grads=get_bool("ac_grads", False),
        reward_grad=get_bool("reward_grad", True),
        repval_loss=get_bool("repval_loss", True),
        repval_grad=get_bool("repval_grad", True),
        report=get_bool("report", True),
        report_gradnorms=get_bool("report_gradnorms", False),
        model_lr=get_float("model_lr", 4e-5),
        actor_lr=get_float("actor_lr", 4e-5),
        critic_lr=get_float("critic_lr", 4e-5),
        opt_agc=get_float("opt_agc", 0.3),
        opt_eps=get_float("opt_eps", 1e-20),
        opt_beta1=get_float("opt_beta1", 0.9),
        opt_beta2=get_float("opt_beta2", 0.999),
        opt_momentum=get_bool("opt_momentum", True),
        opt_nesterov=get_bool("opt_nesterov", False),
        opt_wd=get_float("opt_wd", 0.0),
        opt_wdregex=get_str("opt_wdregex", r"/kernel$"),
        opt_schedule=get_str("opt_schedule", "const"),
        opt_warmup=get_int("opt_warmup", 1000),
        opt_anneal=get_int("opt_anneal", 0),
        actor_min_std=get_float("actor_min_std", 0.1),
        actor_max_std=get_float("actor_max_std", 1.0),
        actor_outscale=get_float("actor_outscale", 0.01),
        rssm_outscale=get_float("rssm_outscale", 1.0),
        reward_outscale=get_float("reward_outscale", 0.0),
        continue_outscale=get_float("continue_outscale", 1.0),
        value_outscale=get_float("value_outscale", 0.0),
        reward_bins=get_int("reward_bins", 255),
        reward_log_low=get_float("reward_log_low", -20.0),
        reward_log_high=get_float("reward_log_high", 20.0),
        beta_pred=get_float("beta_pred", 1.0),
        beta_dyn=get_float("beta_dyn", 1.0),
        beta_rep=get_float("beta_rep", 0.1),
        free_nats=get_float("free_nats", 1.0),
        actent=get_float("actent", 3e-4),
        lambda_=get_float("lambda", 0.95),
        slowreg=get_float("slowreg", 1.0),
        slowtar=get_bool("slowtar", False),
        repval_scale=get_float("repval_scale", 0.3),
        slowvalue_rate=get_float("slowvalue_rate", 0.02),
        slowvalue_every=get_int("slowvalue_every", 1),
        retnorm_impl=get_str("retnorm_impl", "perc"),
        retnorm_rate=get_float("retnorm_rate", 0.01),
        retnorm_limit=get_float("retnorm_limit", 1.0),
        retnorm_perclo=get_float("retnorm_perclo", 5.0),
        retnorm_perchi=get_float("retnorm_perchi", 95.0),
        retnorm_debias=get_bool("retnorm_debias", False),
        valnorm_impl=get_str("valnorm_impl", "none"),
        valnorm_rate=get_float("valnorm_rate", 0.01),
        valnorm_limit=get_float("valnorm_limit", 1e-8),
        valnorm_perclo=get_float("valnorm_perclo", 5.0),
        valnorm_perchi=get_float("valnorm_perchi", 95.0),
        valnorm_debias=get_bool("valnorm_debias", True),
        advnorm_impl=get_str("advnorm_impl", "none"),
        advnorm_rate=get_float("advnorm_rate", 0.01),
        advnorm_limit=get_float("advnorm_limit", 1e-8),
        advnorm_perclo=get_float("advnorm_perclo", 5.0),
        advnorm_perchi=get_float("advnorm_perchi", 95.0),
        advnorm_debias=get_bool("advnorm_debias", True),
    )
