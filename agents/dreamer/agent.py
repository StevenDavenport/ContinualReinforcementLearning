from __future__ import annotations

import json
import random
from contextlib import nullcontext
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from crlbench.core.types import Transition

from .config import DreamerConfig
from .losses import Normalize, imag_loss, replay_value_loss, world_model_loss
from .networks.actor import Actor
from .networks.critic import Critic, SlowCritic
from .networks.decoder import Decoder
from .networks.encoder import Encoder
from .networks.heads import ContinueHead, ContinueHeadConfig, RewardHead, RewardHeadConfig
from .networks.rssm import RSSM, RSSMState
from .optim import LaProp
from .replay import ReplayBatch, SequenceReplayBuffer
from .utils import (
    action_to_model_vector,
    batch_observations_to_model_input,
    env_action_from_model_tensor,
    single_observation_to_model_input,
)


class DreamerAgent:
    def __init__(self, config: DreamerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is unavailable.")

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.compute_dtype = config.torch_dtype
        self._autocast_enabled = (
            config.compute_dtype != "float32"
            and self.device.type in {"cuda", "cpu"}
            and not (self.device.type == "cpu" and self.compute_dtype == torch.float16)
        )

        self.encoder = Encoder(config.build_encoder_config()).to(self.device)
        self.rssm = RSSM(config.build_rssm_config()).to(self.device)
        self.decoder = Decoder(config.build_decoder_config()).to(self.device)
        self.reward_head = RewardHead(
            RewardHeadConfig(
                feat_dim=config.feat_dim,
                hidden_dim=config.model_dim,
                bins=config.reward_bins,
                log_low=config.reward_log_low,
                log_high=config.reward_log_high,
                outscale=config.reward_outscale,
            )
        ).to(self.device)
        self.continue_head = ContinueHead(
            ContinueHeadConfig(
                feat_dim=config.feat_dim,
                hidden_dim=config.model_dim,
                outscale=config.continue_outscale,
            )
        ).to(self.device)
        self.actor = Actor(config.build_actor_config()).to(self.device)
        self.critic = Critic(config.build_critic_config()).to(self.device)
        self.slow_critic = SlowCritic(
            self.critic,
            rate=config.slowvalue_rate,
            every=config.slowvalue_every,
        )

        self.world_model_loss_config = config.build_world_model_loss_config()
        self.actor_loss_config = config.build_actor_loss_config()
        self.critic_loss_config = config.build_critic_loss_config()

        self.retnorm = Normalize(config.build_retnorm_config())
        self.valnorm = Normalize(config.build_valnorm_config())
        self.advnorm = Normalize(config.build_advnorm_config())

        self.world_optimizer = LaProp(
            self._iter_named_world_params(),
            config.build_model_opt_config(),
        )
        self.actor_optimizer = LaProp(
            self._iter_named_params("actor", self.actor),
            config.build_actor_opt_config(),
        )
        self.critic_optimizer = LaProp(
            self._iter_named_params("critic", self.critic),
            config.build_critic_opt_config(),
        )

        self.replay = SequenceReplayBuffer(
            capacity=config.replay_capacity,
            seed=config.seed,
            online=config.replay_online,
            uniform_frac=config.replay_uniform_frac,
            recency_frac=config.replay_recency_frac,
            recency_exp=config.replay_recency_exp,
            chunksize=config.replay_chunksize,
        )

        self._env_steps = 0
        self._updates = 0
        self._train_credit = 0.0

        self._act_state: RSSMState | None = None
        self._act_prev_action: torch.Tensor | None = None
        self._act_is_first = True

    def _iter_named_params(
        self,
        prefix: str,
        module: torch.nn.Module,
    ) -> Iterable[tuple[str, torch.nn.Parameter]]:
        for name, param in module.named_parameters():
            yield f"{prefix}/{name}", param

    def _iter_named_world_params(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        for item in self._iter_named_params("enc", self.encoder):
            yield item
        for item in self._iter_named_params("dyn", self.rssm):
            yield item
        for item in self._iter_named_params("dec", self.decoder):
            yield item
        for item in self._iter_named_params("rew", self.reward_head):
            yield item
        for item in self._iter_named_params("con", self.continue_head):
            yield item

    def reset(self) -> None:
        self._act_state = None
        self._act_prev_action = None
        self._act_is_first = True

    def _autocast(self):
        if self._autocast_enabled:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
                enabled=True,
            )
        return nullcontext()

    @torch.no_grad()
    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False) -> object:
        obs = single_observation_to_model_input(
            observation,
            vector_dim=self.config.vector_input_dim,
            device=self.device,
        )
        with self._autocast():
            embed = self.encoder(obs)
        if self._act_state is None:
            self._act_state = self.rssm.initial(1, device=self.device, dtype=embed.dtype)
        if self._act_prev_action is None:
            self._act_prev_action = torch.zeros(
                1,
                self.config.action_dim,
                device=self.device,
                dtype=embed.dtype,
            )
        is_first = torch.tensor(
            [1.0 if self._act_is_first else 0.0],
            device=self.device,
            dtype=embed.dtype,
        )
        with self._autocast():
            post, _prior, _post_dist, _prior_dist = self.rssm.obs_step(
                self._act_state,
                self._act_prev_action,
                embed,
                is_first=is_first,
                sample=not deterministic,
            )
            feat = self.rssm.get_feat(post)
            terms = self.actor.policy_terms(feat, deterministic=deterministic, straight_through=False)

        env_action_tensor = self.actor.to_env_action(terms["action"])
        env_action = env_action_from_model_tensor(
            env_action_tensor.detach().cpu(),
            action_space=self.config.action_space,
        )

        self._act_state = RSSM.detach_state(post)
        self._act_prev_action = terms["model_action"].detach()
        self._act_is_first = False
        return env_action

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]:
        context_payload: dict[str, torch.Tensor] | None = None
        if len(batch) == 1 and self._act_state is not None:
            context_payload = {
                "deter": self._act_state.deter[0].detach().to(dtype=torch.float32).cpu(),
                "stoch": self._act_state.stoch[0].detach().to(dtype=torch.float32).cpu(),
                "logits": self._act_state.logits[0].detach().to(dtype=torch.float32).cpu(),
            }

        for transition in batch:
            done = bool(transition.terminated or transition.truncated)
            cont = 0.0 if done else 1.0
            action_vector = action_to_model_vector(
                transition.action,
                action_space=self.config.action_space,
                action_dim=self.config.action_dim,
            )
            self.replay.add(
                observation=dict(transition.observation),
                action=action_vector,
                reward=float(transition.reward),
                cont=cont,
                done=done,
                is_terminal=bool(transition.terminated),
                context=context_payload,
            )
            self._env_steps += 1
            if done:
                self._act_is_first = True

        metrics = self._empty_metrics()
        if self._env_steps < self.config.warmup_steps:
            return metrics

        self._train_credit += float(len(batch)) * self.config.train_updates_per_env_step
        updates = int(self._train_credit)
        if updates <= 0:
            return metrics
        self._train_credit -= float(updates)

        aggregate: dict[str, float] = {}
        ran = 0
        for _ in range(updates):
            sampled = self.replay.sample(
                batch_size=self.config.batch_size,
                sequence_length=self.config.batch_length,
            )
            if sampled is None:
                break
            train_metrics = self._train_step(sampled)
            ran += 1
            for key, value in train_metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)

        if ran == 0:
            return metrics
        for key, value in aggregate.items():
            metrics[key] = value / float(ran)
        metrics["num_updates"] = float(ran)
        self._updates += ran
        return metrics

    def _empty_metrics(self) -> dict[str, float]:
        metrics = {
            "world_model_total": 0.0,
            "pred_total": 0.0,
            "pred_image": 0.0,
            "pred_vector": 0.0,
            "pred_reward": 0.0,
            "pred_continue": 0.0,
            "kl_dyn": 0.0,
            "kl_rep": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "repval_loss": 0.0,
            "loss_total": 0.0,
            "imag_return_mean": 0.0,
            "num_updates": 0.0,
        }
        if self.config.report_gradnorms:
            metrics.update(
                {
                    "gradnorm_world": 0.0,
                    "gradnorm_actor": 0.0,
                    "gradnorm_critic": 0.0,
                }
            )
        return metrics

    @staticmethod
    def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        sq_sum = None
        for param in parameters:
            grad = param.grad
            if grad is None:
                continue
            term = grad.detach().float().pow(2).sum()
            sq_sum = term if sq_sum is None else (sq_sum + term)
        if sq_sum is None:
            return torch.tensor(0.0)
        return sq_sum.sqrt()

    def _to_batch_major(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(f"Expected tensor rank >=2 with leading [T,B], got {tensor.shape}.")
        perm = [1, 0, *range(2, tensor.ndim)]
        return tensor.permute(*perm)

    def _select_imag_starts(self, posterior: RSSMState, *, k: int) -> RSSMState:
        deter = self._to_batch_major(posterior.deter)[..., -k:, :]
        stoch = self._to_batch_major(posterior.stoch)[..., -k:, :, :]
        logits = self._to_batch_major(posterior.logits)[..., -k:, :, :]

        batch = int(deter.shape[0])
        return RSSMState(
            deter=deter.reshape(batch * k, deter.shape[-1]),
            stoch=stoch.reshape(batch * k, stoch.shape[-2], stoch.shape[-1]),
            logits=logits.reshape(batch * k, logits.shape[-2], logits.shape[-1]),
        )

    def _imagine(
        self,
        start: RSSMState,
        *,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")

        state = start
        feats: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for _ in range(horizon):
            feat = self.rssm.get_feat(state)
            terms = self.actor.policy_terms(feat.detach(), deterministic=False, straight_through=False)
            action_model = terms["model_action"]

            next_state, _prior_dist = self.rssm.img_step(state, action_model.detach(), sample=True)
            next_feat = self.rssm.get_feat(next_state)

            feats.append(next_feat)
            actions.append(action_model)
            log_probs.append(terms["log_prob"])
            entropies.append(terms["entropy"])
            state = next_state

        return {
            "feat": torch.stack(feats, dim=1),
            "action": torch.stack(actions, dim=1),
            "log_prob": torch.stack(log_probs, dim=1),
            "entropy": torch.stack(entropies, dim=1),
        }

    def _train_step(self, batch: ReplayBatch) -> dict[str, float]:
        self.encoder.train(True)
        self.rssm.train(True)
        self.decoder.train(True)
        self.reward_head.train(True)
        self.continue_head.train(True)
        self.actor.train(True)
        self.critic.train(True)

        observations = batch_observations_to_model_input(
            batch.observations,
            vector_dim=self.config.vector_input_dim,
            device=self.device,
        )
        actions = batch.actions.to(device=self.device, dtype=torch.float32)
        prev_actions = batch.prev_actions.to(device=self.device, dtype=torch.float32)
        rewards = batch.rewards.to(device=self.device, dtype=torch.float32)
        continues = batch.continues.to(device=self.device, dtype=torch.float32)
        is_first = batch.is_first.to(device=self.device, dtype=torch.float32)
        is_last = batch.is_last.to(device=self.device, dtype=torch.float32)
        is_terminal = batch.is_terminal.to(device=self.device, dtype=torch.float32)
        init_deter = batch.init_deter.to(device=self.device, dtype=torch.float32)
        init_stoch = batch.init_stoch.to(device=self.device, dtype=torch.float32)
        init_logits = batch.init_logits.to(device=self.device, dtype=torch.float32)
        init_mask = batch.init_mask.to(device=self.device, dtype=torch.float32)

        if actions.ndim != 3:
            raise ValueError(f"Expected actions shape [T,B,A], got {tuple(actions.shape)}.")
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"Expected action dim {self.config.action_dim}, got {actions.shape[-1]}."
            )

        init_state: RSSMState | None = None
        if init_deter.shape[-1] > 0:
            init_state = RSSMState(
                deter=init_deter * init_mask.unsqueeze(-1),
                stoch=init_stoch * init_mask[:, None, None],
                logits=init_logits * init_mask[:, None, None],
            )

        with self._autocast():
            embed = self.encoder(observations)
            posterior, prior, _ = self.rssm.observe(
                embed,
                prev_actions,
                is_first,
                state=init_state,
                sample=True,
            )
            repfeat = self.rssm.get_feat(posterior)

            decoded = self.decoder(repfeat)
            targets = self.decoder.preprocess_targets(observations)

            rew_inp = repfeat if self.config.reward_grad else repfeat.detach()
            reward_logits = self.reward_head(rew_inp)
            continue_logits = self.continue_head(repfeat)
        continue_target = continues
        if self.config.contdisc:
            continue_target = continue_target * (1.0 - (1.0 / float(self.config.horizon)))

        world_metrics = world_model_loss(
            decoded=decoded,
            targets=targets,
            reward_logits=reward_logits,
            reward_target=rewards,
            continue_logits=continue_logits,
            continue_target=continue_target,
            posterior_logits=posterior.logits,
            prior_logits=prior.logits,
            config=self.world_model_loss_config,
        )

        batch_size = int(actions.shape[1])
        time_steps = int(actions.shape[0])
        k = min(self.config.imag_last or time_steps, time_steps)

        starts = self._select_imag_starts(RSSM.detach_state(posterior), k=k)
        with self._autocast():
            imagined = self._imagine(starts, horizon=self.config.imagine_horizon)

        repfeat_btk = self._to_batch_major(repfeat)
        first = repfeat_btk[:, -k:, :].reshape(batch_size * k, 1, repfeat_btk.shape[-1])
        first = first if self.config.ac_grads else first.detach()

        imgfeat = imagined["feat"] if self.config.ac_grads else imagined["feat"].detach()
        imgfeat_all = torch.cat([first, imgfeat], dim=1)

        with self._autocast():
            last_terms = self.actor.policy_terms(
                imgfeat_all[:, -1].detach(),
                deterministic=False,
                straight_through=False,
            )
        imgact = torch.cat([imagined["action"], last_terms["model_action"][:, None]], dim=1)
        imglogp = torch.cat([imagined["log_prob"], last_terms["log_prob"][:, None]], dim=1)
        imgent = torch.cat([imagined["entropy"], last_terms["entropy"][:, None]], dim=1)

        with self._autocast():
            rew = self.reward_head.predict(imgfeat_all)
            con = torch.sigmoid(self.continue_head(imgfeat_all))
            value_logits = self.critic(imgfeat_all)
            slowvalue_logits = self.slow_critic.logits(imgfeat_all)

        imag_losses, imag_outs, imag_metrics = imag_loss(
            act_logprob=imglogp,
            act_entropy=imgent,
            act_sample=imgact,
            rew=rew,
            con=con,
            value_logits=value_logits,
            slowvalue_logits=slowvalue_logits,
            retnorm=self.retnorm,
            valnorm=self.valnorm,
            advnorm=self.advnorm,
            actor_config=self.actor_loss_config,
            critic_config=self.critic_loss_config,
            update=True,
        )

        repval_loss_mean = torch.zeros((), device=self.device, dtype=rew.dtype)
        if self.config.repval_loss:
            replay_feat = repfeat_btk if self.config.repval_grad else repfeat_btk.detach()
            replay_feat = replay_feat[:, -k:, :]
            replay_last = self._to_batch_major(is_last)[:, -k:]
            replay_term = self._to_batch_major(is_terminal)[:, -k:]
            replay_rew = self._to_batch_major(rewards)[:, -k:]
            boot = imag_outs["ret"][:, 0].reshape(batch_size, k)

            with self._autocast():
                replay_value_logits = self.critic(replay_feat)
                replay_slow_logits = self.slow_critic.logits(replay_feat)
            repl_losses, _repl_outs, _repl_metrics = replay_value_loss(
                last=replay_last,
                term=replay_term,
                rew=replay_rew,
                boot=boot,
                value_logits=replay_value_logits,
                slowvalue_logits=replay_slow_logits,
                valnorm=self.valnorm,
                config=self.critic_loss_config,
                update=True,
            )
            repval_loss_mean = repl_losses["repval"].mean()

        policy_loss = imag_losses["policy"].mean()
        value_loss = imag_losses["value"].mean()

        total_loss = (
            world_metrics["world_model_total"]
            + policy_loss
            + value_loss
            + (self.critic_loss_config.repval_scale * repval_loss_mean)
        )

        self.world_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        gradnorm_world = torch.tensor(0.0, device=total_loss.device)
        gradnorm_actor = torch.tensor(0.0, device=total_loss.device)
        gradnorm_critic = torch.tensor(0.0, device=total_loss.device)
        if self.config.report_gradnorms:
            gradnorm_world = self._grad_norm(self.world_optimizer.param_groups[0]["params"]).to(
                device=total_loss.device
            )
            gradnorm_actor = self._grad_norm(self.actor_optimizer.param_groups[0]["params"]).to(
                device=total_loss.device
            )
            gradnorm_critic = self._grad_norm(
                self.critic_optimizer.param_groups[0]["params"]
            ).to(device=total_loss.device)

        self.world_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.slow_critic.update(self.critic)

        metrics = {
            "world_model_total": float(world_metrics["world_model_total"].detach().item()),
            "pred_total": float(world_metrics["pred_total"].detach().item()),
            "pred_image": float(world_metrics["pred_image"].detach().item()),
            "pred_vector": float(world_metrics["pred_vector"].detach().item()),
            "pred_reward": float(world_metrics["pred_reward"].detach().item()),
            "pred_continue": float(world_metrics["pred_continue"].detach().item()),
            "kl_dyn": float(world_metrics["kl_dyn"].detach().item()),
            "kl_rep": float(world_metrics["kl_rep"].detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "repval_loss": float(repval_loss_mean.detach().item()),
            "loss_total": float(total_loss.detach().item()),
            "imag_return_mean": float(imag_outs["ret"].mean().detach().item()),
            "imag_entropy": float(imag_metrics["entropy"].detach().item()),
        }
        if self.config.report_gradnorms:
            metrics.update(
                {
                    "gradnorm_world": float(gradnorm_world.detach().item()),
                    "gradnorm_actor": float(gradnorm_actor.detach().item()),
                    "gradnorm_critic": float(gradnorm_critic.detach().item()),
                }
            )
        return metrics

    @torch.no_grad()
    def report_openloop(
        self,
        batch: ReplayBatch,
        *,
        max_batch: int = 6,
    ) -> dict[str, torch.Tensor]:
        if not self.config.report:
            return {}

        observations = batch_observations_to_model_input(
            batch.observations,
            vector_dim=self.config.vector_input_dim,
            device=self.device,
        )
        prev_actions = batch.prev_actions.to(device=self.device, dtype=torch.float32)
        is_first = batch.is_first.to(device=self.device, dtype=torch.float32)

        time_steps = int(prev_actions.shape[0])
        if time_steps < 2:
            return {}
        split = time_steps // 2
        rb = min(max_batch, int(prev_actions.shape[1]))

        obs_half = {k: v[:split, :rb] for k, v in observations.items()}
        prev_half = prev_actions[:split, :rb]
        first_half = is_first[:split, :rb]

        with self._autocast():
            embed_half = self.encoder(obs_half)
            post_half, _prior_half, carry = self.rssm.observe(
                embed_half,
                prev_half,
                first_half,
                sample=False,
            )
            obs_feat = self.rssm.get_feat(post_half)

            imag_actions = prev_actions[split:, :rb]
            imag_states, _ = self.rssm.imagine(imag_actions, state=carry, sample=False)
            imag_feat = self.rssm.get_feat(imag_states)

            obs_recons = self.decoder(obs_feat)
            img_recons = self.decoder(imag_feat)

        reports: dict[str, torch.Tensor] = {}
        if "pixels" in obs_recons and "pixels" in img_recons and "pixels" in observations:
            true = observations["pixels"][:time_steps, :rb].to(dtype=obs_recons["pixels"].dtype)
            if true.max().item() > 1.0:
                true = true / 255.0
            pred = torch.cat([obs_recons["pixels"], img_recons["pixels"]], dim=0)
            pred = pred[:time_steps]
            error = (pred - true).abs().clamp(0.0, 1.0)
            reports["openloop/pixels_true"] = true.detach().cpu()
            reports["openloop/pixels_pred"] = pred.detach().cpu()
            reports["openloop/pixels_error"] = error.detach().cpu()
        return reports

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = path.parent / f"{path.stem}.pt"
        torch.save(
            {
                "config": self.config.__dict__,
                "encoder": self.encoder.state_dict(),
                "rssm": self.rssm.state_dict(),
                "decoder": self.decoder.state_dict(),
                "reward_head": self.reward_head.state_dict(),
                "continue_head": self.continue_head.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "slow_critic": self.slow_critic.model.state_dict(),
                "slow_critic_count": self.slow_critic.count,
                "optimizers": {
                    "world": self.world_optimizer.state_dict(),
                    "actor": self.actor_optimizer.state_dict(),
                    "critic": self.critic_optimizer.state_dict(),
                },
                "counters": {
                    "env_steps": self._env_steps,
                    "updates": self._updates,
                    "train_credit": self._train_credit,
                    "replay_size": self.replay.size,
                },
            },
            weights_path,
        )
        payload = {
            "agent": "dreamer",
            "version": "0.3.0",
            "weights": weights_path.name,
            "env_steps": self._env_steps,
            "updates": self._updates,
            "replay_size": self.replay.size,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
