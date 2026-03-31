from __future__ import annotations

import sys
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from dreamer.losses import lambda_return
from dreamer.networks.distributions import (
    symexp,
    symexp_twohot_bins,
    symlog,
    twohot_cross_entropy,
    twohot_mean,
)
from dreamer_jax_phase1_reference import (
    lambda_return as jax_lambda_return,
)
from dreamer_jax_phase1_reference import (
    original_symexp_twohot_bins,
    runtime_label,
    symexp as jax_symexp,
    symlog as jax_symlog,
    twohot_loss,
    twohot_pred,
)


def test_phase1_runtime_label_is_known() -> None:
    assert runtime_label() in {"jax-runtime", "reference-formulas"}


def test_default_twohot_support_matches_original_odd_bins() -> None:
    torch_bins = symexp_twohot_bins(bins=255)
    ref_bins = torch.from_numpy(original_symexp_twohot_bins(255))
    torch.testing.assert_close(torch_bins, ref_bins, rtol=2e-6, atol=5.0)


def test_symlog_symexp_parity_against_original_formulas() -> None:
    values = torch.tensor(
        [-20.0, -8.0, -1.0, -1.0e-3, 0.0, 1.0e-3, 1.0, 8.0, 20.0],
        dtype=torch.float32,
    )
    np_values = values.numpy()

    torch.testing.assert_close(symlog(values), torch.from_numpy(jax_symlog(np_values)), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(symexp(values), torch.from_numpy(jax_symexp(np_values)), rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_twohot_mean_parity_against_original_formulas() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 5, 255, generator=generator)
    support = torch.from_numpy(original_symexp_twohot_bins(255))

    actual = twohot_mean(logits, support=support)
    expected = torch.from_numpy(twohot_pred(logits.numpy(), support.numpy()))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@torch.no_grad()
def test_twohot_cross_entropy_parity_against_original_formulas() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(4, 6, 255, generator=generator)
    target = torch.linspace(-4.0, 4.0, steps=24, dtype=torch.float32).reshape(4, 6)
    support = torch.from_numpy(original_symexp_twohot_bins(255))

    actual = twohot_cross_entropy(logits, target, support=support)
    expected = torch.from_numpy(twohot_loss(logits.numpy(), target.numpy(), support.numpy()))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@torch.no_grad()
def test_lambda_return_parity_against_original_formulas() -> None:
    generator = torch.Generator().manual_seed(23)
    batch = 3
    time = 6
    last = torch.zeros(batch, time)
    term = torch.zeros(batch, time)
    term[1, 4] = 1.0
    rew = torch.randn(batch, time, generator=generator)
    val = torch.randn(batch, time, generator=generator)
    boot = torch.randn(batch, time, generator=generator)

    actual = lambda_return(last, term, rew, val, boot, disc=0.997, lam=0.95)
    expected = torch.from_numpy(
        jax_lambda_return(
            last.numpy(),
            term.numpy(),
            rew.numpy(),
            val.numpy(),
            boot.numpy(),
            disc=0.997,
            lam=0.95,
        )
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
