from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crlbench.core.types import Transition


def _flatten_numeric(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Mapping):
        out: list[float] = []
        for key in sorted(value):
            out.extend(_flatten_numeric(value[key]))
        return out
    if isinstance(value, list | tuple):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return [1.0]
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    if total <= 0:
        return [1.0 / len(exp_values) for _ in exp_values]
    return [value / total for value in exp_values]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PPOBaselineAgent:
    def __init__(  # noqa: PLR0913
        self,
        *,
        seed: int = 0,
        action_space_n: int = 2,
        learning_rate: float = 0.01,
        value_learning_rate: float = 0.02,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> None:
        if action_space_n <= 1:
            raise ValueError(f"action_space_n must be > 1, got {action_space_n}.")
        self.seed = seed
        self.learning_rate = learning_rate
        self.value_learning_rate = value_learning_rate
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self._default_action_space_n = action_space_n
        self._rng = random.Random(seed)

        self._policy_weights: list[list[float]] = [[] for _ in range(action_space_n)]
        self._policy_bias: list[float] = [0.0 for _ in range(action_space_n)]
        self._value_weights: list[float] = []
        self._value_bias = 0.0

        self._last_action: int | None = None
        self._last_action_prob: float | None = None

    def _features(self, observation: Mapping[str, Any]) -> list[float]:
        features = _flatten_numeric(observation)
        return features if features else [0.0]

    def _action_space_n(self, observation: Mapping[str, Any]) -> int:
        value = observation.get("action_space_n", self._default_action_space_n)
        if not isinstance(value, int) or value <= 1:
            return self._default_action_space_n
        return value

    def _ensure_shape(self, *, feature_count: int, action_space_n: int) -> None:
        if len(self._policy_weights) < action_space_n:
            for _ in range(action_space_n - len(self._policy_weights)):
                self._policy_weights.append([0.0 for _ in range(feature_count)])
                self._policy_bias.append(0.0)

        for action in range(action_space_n):
            weights = self._policy_weights[action]
            if len(weights) < feature_count:
                weights.extend([0.0] * (feature_count - len(weights)))
            elif len(weights) > feature_count:
                del weights[feature_count:]

        if len(self._value_weights) < feature_count:
            self._value_weights.extend([0.0] * (feature_count - len(self._value_weights)))
        elif len(self._value_weights) > feature_count:
            del self._value_weights[feature_count:]

    def _policy_probs(self, features: list[float], action_space_n: int) -> list[float]:
        logits: list[float] = []
        for action in range(action_space_n):
            logit = self._policy_bias[action]
            weights = self._policy_weights[action]
            for weight, feature in zip(weights, features, strict=True):
                logit += weight * feature
            logits.append(logit)
        return _softmax(logits)

    def _value(self, features: list[float]) -> float:
        value = self._value_bias
        for weight, feature in zip(self._value_weights, features, strict=True):
            value += weight * feature
        return value

    def reset(self) -> None:
        self._last_action = None
        self._last_action_prob = None

    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False) -> int:
        features = self._features(observation)
        action_space_n = self._action_space_n(observation)
        self._ensure_shape(feature_count=len(features), action_space_n=action_space_n)
        probs = self._policy_probs(features, action_space_n)
        if deterministic:
            action = max(range(action_space_n), key=lambda index: probs[index])
        else:
            threshold = self._rng.random()
            cumulative = 0.0
            action = action_space_n - 1
            for index, probability in enumerate(probs):
                cumulative += probability
                if threshold <= cumulative:
                    action = index
                    break
        self._last_action = action
        self._last_action_prob = probs[action]
        return action

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]:
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "num_updates": 0.0}

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for transition in batch:
            observation = transition.observation
            features = self._features(observation)
            action_space_n = self._action_space_n(observation)
            self._ensure_shape(feature_count=len(features), action_space_n=action_space_n)
            probs = self._policy_probs(features, action_space_n)
            value = self._value(features)
            action = int(transition.action) if isinstance(transition.action, int) else 0
            action = max(0, min(action_space_n - 1, action))

            reward = float(transition.reward)
            advantage = _clamp(reward - value, -5.0, 5.0)
            old_prob = self._last_action_prob if self._last_action == action else probs[action]
            safe_old_prob = max(1e-6, min(1.0 - 1e-6, old_prob))
            safe_prob = max(1e-6, min(1.0 - 1e-6, probs[action]))

            ratio = safe_prob / safe_old_prob
            clipped_ratio = min(1.0 + self.clip_epsilon, max(1.0 - self.clip_epsilon, ratio))
            coeff = _clamp(clipped_ratio * advantage, -5.0, 5.0)

            for index in range(action_space_n):
                indicator = 1.0 if index == action else 0.0
                grad_logits = _clamp((indicator - probs[index]) * coeff, -5.0, 5.0)
                weights = self._policy_weights[index]
                for feat_index, feature in enumerate(features):
                    update = self.learning_rate * grad_logits * feature
                    weights[feat_index] = _clamp(weights[feat_index] + update, -10.0, 10.0)
                # Encourage exploration via small entropy-style uniform pull.
                weights_entropy = ((1.0 / action_space_n) - probs[index]) * self.entropy_coef
                bias_update = self.learning_rate * (grad_logits + weights_entropy)
                self._policy_bias[index] = _clamp(
                    self._policy_bias[index] + bias_update,
                    -10.0,
                    10.0,
                )

            for feat_index, feature in enumerate(features):
                value_update = self.value_learning_rate * advantage * feature
                self._value_weights[feat_index] = _clamp(
                    self._value_weights[feat_index] + value_update,
                    -10.0,
                    10.0,
                )
            self._value_bias = _clamp(
                self._value_bias + (self.value_learning_rate * advantage),
                -10.0,
                10.0,
            )

            policy_losses.append(-math.log(safe_prob) * advantage)
            value_losses.append(0.5 * (reward - value) ** 2)
            entropy = -sum(probability * math.log(max(1e-6, probability)) for probability in probs)
            entropies.append(entropy)

        self._last_action = None
        self._last_action_prob = None
        count = float(len(batch))
        return {
            "policy_loss": float(sum(policy_losses) / count),
            "value_loss": float(sum(value_losses) / count),
            "entropy": float(sum(entropies) / count),
            "num_updates": count,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "value_learning_rate": self.value_learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "default_action_space_n": self._default_action_space_n,
            "policy_weights": self._policy_weights,
            "policy_bias": self._policy_bias,
            "value_weights": self._value_weights,
            "value_bias": self._value_bias,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_agent(config: Mapping[str, Any]) -> PPOBaselineAgent:
    seed_raw = config.get("seed", 0)
    action_space_n_raw = config.get("action_space_n", 2)
    learning_rate_raw = config.get("learning_rate", 0.01)
    value_learning_rate_raw = config.get("value_learning_rate", 0.02)
    clip_epsilon_raw = config.get("clip_epsilon", 0.2)
    entropy_coef_raw = config.get("entropy_coef", 0.01)
    if not isinstance(seed_raw, int):
        raise ValueError(f"seed must be int, got {seed_raw!r}.")
    if not isinstance(action_space_n_raw, int):
        raise ValueError(f"action_space_n must be int, got {action_space_n_raw!r}.")
    if not isinstance(learning_rate_raw, int | float):
        raise ValueError(f"learning_rate must be numeric, got {learning_rate_raw!r}.")
    if not isinstance(value_learning_rate_raw, int | float):
        raise ValueError(f"value_learning_rate must be numeric, got {value_learning_rate_raw!r}.")
    if not isinstance(clip_epsilon_raw, int | float):
        raise ValueError(f"clip_epsilon must be numeric, got {clip_epsilon_raw!r}.")
    if not isinstance(entropy_coef_raw, int | float):
        raise ValueError(f"entropy_coef must be numeric, got {entropy_coef_raw!r}.")
    return PPOBaselineAgent(
        seed=seed_raw,
        action_space_n=action_space_n_raw,
        learning_rate=float(learning_rate_raw),
        value_learning_rate=float(value_learning_rate_raw),
        clip_epsilon=float(clip_epsilon_raw),
        entropy_coef=float(entropy_coef_raw),
    )
