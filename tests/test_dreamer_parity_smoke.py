from __future__ import annotations

import sys
from pathlib import Path

import torch

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from dreamer.losses import (
    ActorLossConfig,
    CriticLossConfig,
    Normalize,
    NormalizeConfig,
    WorldModelLossConfig,
    imag_loss,
    replay_value_loss,
    world_model_loss,
)
from dreamer.networks.actor import Actor, ActorConfig
from dreamer.optim import LaProp, LaPropConfig


def _norm_none() -> Normalize:
    return Normalize(NormalizeConfig(impl="none"))


def test_world_model_loss_smoke() -> None:
    batch = 2
    time = 4
    bins = 255

    decoded = {
        "pixels": torch.rand(time, batch, 8, 8, 3),
        "proprio": torch.rand(time, batch, 5),
    }
    targets = {
        "pixels": torch.rand(time, batch, 8, 8, 3),
        "proprio": torch.rand(time, batch, 5),
    }

    metrics = world_model_loss(
        decoded=decoded,
        targets=targets,
        reward_logits=torch.randn(time, batch, bins),
        reward_target=torch.randn(time, batch),
        continue_logits=torch.randn(time, batch),
        continue_target=torch.rand(time, batch),
        posterior_logits=torch.randn(time, batch, 8, 8),
        prior_logits=torch.randn(time, batch, 8, 8),
        config=WorldModelLossConfig(),
    )
    assert torch.isfinite(metrics["world_model_total"])
    assert metrics["pred_image"].ndim == 0


def test_imag_and_replay_losses_smoke() -> None:
    batch = 2
    horizon_plus_1 = 5
    bins = 255

    retnorm = _norm_none()
    valnorm = _norm_none()
    advnorm = _norm_none()

    imag_losses, imag_outs, _ = imag_loss(
        act_logprob=torch.randn(batch, horizon_plus_1),
        act_entropy=torch.rand(batch, horizon_plus_1),
        act_sample=torch.randn(batch, horizon_plus_1, 3),
        rew=torch.randn(batch, horizon_plus_1),
        con=torch.rand(batch, horizon_plus_1),
        value_logits=torch.randn(batch, horizon_plus_1, bins),
        slowvalue_logits=torch.randn(batch, horizon_plus_1, bins),
        retnorm=retnorm,
        valnorm=valnorm,
        advnorm=advnorm,
        actor_config=ActorLossConfig(),
        critic_config=CriticLossConfig(),
        update=True,
    )
    assert imag_losses["policy"].shape == (batch, horizon_plus_1 - 1)
    assert imag_losses["value"].shape == (batch, horizon_plus_1 - 1)
    assert imag_outs["ret"].shape == (batch, horizon_plus_1 - 1)

    repl_losses, repl_outs, _ = replay_value_loss(
        last=torch.zeros(batch, horizon_plus_1),
        term=torch.zeros(batch, horizon_plus_1),
        rew=torch.randn(batch, horizon_plus_1),
        boot=torch.randn(batch, horizon_plus_1),
        value_logits=torch.randn(batch, horizon_plus_1, bins),
        slowvalue_logits=torch.randn(batch, horizon_plus_1, bins),
        valnorm=valnorm,
        config=CriticLossConfig(),
        update=True,
    )
    assert repl_losses["repval"].shape == (batch, horizon_plus_1 - 1)
    assert repl_outs["ret"].shape == (batch, horizon_plus_1 - 1)


def test_laprop_update_smoke() -> None:
    linear = torch.nn.Linear(8, 4)
    opt = LaProp(
        ((f"lin/{name}", param) for name, param in linear.named_parameters()),
        LaPropConfig(lr=1e-3, agc=0.3),
    )
    x = torch.randn(6, 8)
    y = linear(x).pow(2).mean()
    y.backward()
    before = {name: param.detach().clone() for name, param in linear.named_parameters()}
    opt.step()

    changed = []
    for name, param in linear.named_parameters():
        changed.append(bool((before[name] - param.detach()).abs().sum() > 0))
        assert torch.isfinite(param).all()
    assert any(changed)


def test_actor_distribution_smoke() -> None:
    feat = torch.randn(3, 32)

    disc_actor = Actor(
        ActorConfig(
            feat_dim=32,
            hidden_dim=64,
            hidden_layers=2,
            action_dim=5,
            action_space="discrete",
        )
    )
    disc_terms = disc_actor.policy_terms(feat, deterministic=False, straight_through=False)
    assert disc_terms["action"].shape == (3,)
    assert disc_terms["model_action"].shape == (3, 5)
    assert disc_terms["log_prob"].shape == (3,)

    cont_actor = Actor(
        ActorConfig(
            feat_dim=32,
            hidden_dim=64,
            hidden_layers=2,
            action_dim=4,
            action_space="continuous",
        )
    )
    cont_terms = cont_actor.policy_terms(feat, deterministic=False, straight_through=False)
    assert cont_terms["action"].shape == (3, 4)
    assert cont_terms["log_prob"].shape == (3,)
    assert torch.isfinite(cont_terms["entropy"]).all()
