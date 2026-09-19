from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        log_std_init: float = -0.35,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)


class SegmentAttentionActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        segment_count: int = 30,
        maturity_bucket_count: int = 6,
        segment_parameter_count: int = 4,
        global_feature_count: int = 1,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.0,
        log_std_init: float = -0.80,
    ) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.maturity_bucket_count = int(maturity_bucket_count)
        self.segment_parameter_count = int(segment_parameter_count)
        self.global_feature_count = int(global_feature_count)
        expected_obs = (
            self.global_feature_count
            + self.segment_count
            + self.segment_count * self.maturity_bucket_count
            + self.segment_count * self.segment_parameter_count
        )
        expected_action = int(action_dim)
        global_action_count = expected_action - self.segment_count
        if global_action_count < 1:
            raise ValueError(
                f"Attention policy expected at least one global action head, got action_dim={action_dim}"
            )
        if observation_dim != expected_obs:
            raise ValueError(f"Attention policy expected observation_dim={expected_obs}, got {observation_dim}")
        if action_dim != expected_action:
            raise ValueError(f"Attention policy expected action_dim={expected_action}, got {action_dim}")

        token_dim = self.global_feature_count + 1 + self.maturity_bucket_count + self.segment_parameter_count
        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.segment_embedding = nn.Parameter(torch.zeros(self.segment_count, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        self.segment_actor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.deploy_actor = nn.Sequential(
            nn.LayerNorm(hidden_size + self.global_feature_count),
            nn.Linear(hidden_size + self.global_feature_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, global_action_count),
        )
        nn.init.zeros_(self.deploy_actor[-1].weight)
        nn.init.zeros_(self.deploy_actor[-1].bias)
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_size + self.global_feature_count),
            nn.Linear(hidden_size + self.global_feature_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        cursor = 0
        global_features = observations[:, cursor : cursor + self.global_feature_count]
        cursor += self.global_feature_count
        demand = observations[:, cursor : cursor + self.segment_count].reshape(batch, self.segment_count, 1)
        cursor += self.segment_count
        exposure = observations[:, cursor : cursor + self.segment_count * self.maturity_bucket_count].reshape(
            batch, self.segment_count, self.maturity_bucket_count
        )
        cursor += self.segment_count * self.maturity_bucket_count
        params = observations[:, cursor : cursor + self.segment_count * self.segment_parameter_count].reshape(
            batch, self.segment_count, self.segment_parameter_count
        )
        global_token = global_features.unsqueeze(1).expand(-1, self.segment_count, -1)
        tokens = torch.cat([global_token, demand, exposure, params], dim=-1)
        encoded = self.token_projection(tokens) + self.segment_embedding.unsqueeze(0)
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        global_state = torch.cat([pooled, global_features], dim=1)
        return encoded, global_state, global_features

    def distribution(self, observations: torch.Tensor) -> Normal:
        encoded, global_state, _ = self._encode(observations)
        segment_logits = self.segment_actor(encoded).squeeze(-1)
        global_action_logits = self.deploy_actor(global_state)
        mean = torch.cat([global_action_logits, segment_logits], dim=1)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        _, global_state, _ = self._encode(observations)
        return self.critic(global_state).squeeze(-1)


@dataclass
class PPOConfig:
    episodes: int = 300
    rollout_episodes: int = 8
    update_epochs: int = 6
    minibatch_size: int = 1024
    hidden_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.50
    action_clip: float = 10.0
    seed: int = 42
    device: str = "auto"
    architecture: str = "mlp"
    segment_count: int = 30
    maturity_bucket_count: int = 6
    segment_parameter_count: int = 4
    global_feature_count: int = 1
    attention_heads: int = 4
    attention_layers: int = 2
    attention_dropout: float = 0.0
    primal_dual: bool = False
    dual_lambda: float = 0.0
    dual_learning_rate: float = 0.10
    dual_max_lambda: float = 100.0
    constraint_metric: str = "liquidated"
    constraint_threshold: float = 0.01
    teacher_kl_coef: float = 0.0
    teacher_kl_decay: float = 1.0
    teacher_kl_min_coef: float = 0.0
    teacher_kl_stop_update: int = 0
    checkpoint_dir: str | None = None
    checkpoint_every_updates: int = 0
    action_log_std_init: float = -0.80
    freeze_action_std: bool = False
    gate_blend_factors: list[float] | None = None


class PPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = _resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if config.architecture == "attention":
            self.model = SegmentAttentionActorCritic(
                observation_dim,
                action_dim,
                hidden_size=config.hidden_size,
                segment_count=config.segment_count,
                maturity_bucket_count=config.maturity_bucket_count,
                segment_parameter_count=config.segment_parameter_count,
                global_feature_count=config.global_feature_count,
                attention_heads=config.attention_heads,
                attention_layers=config.attention_layers,
                attention_dropout=config.attention_dropout,
                log_std_init=config.action_log_std_init,
            ).to(self.device)
        elif config.architecture == "mlp":
            self.model = ActorCritic(
                observation_dim,
                action_dim,
                hidden_size=config.hidden_size,
                log_std_init=config.action_log_std_init,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown policy architecture: {config.architecture}")
        if bool(config.freeze_action_std):
            self.model.log_std.requires_grad_(False)
        self.configure_optimizer()

    def configure_optimizer(self) -> None:
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("PPOAgent has no trainable parameters.")
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.learning_rate)

    def set_trainable_scope(self, scope: str) -> None:
        scope = str(scope)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if bool(self.config.freeze_action_std):
            self.model.log_std.requires_grad_(False)
        if scope == "all":
            self.configure_optimizer()
            return
        if scope != "deploy_actor_and_critic":
            raise ValueError(f"Unknown PPO trainable scope: {scope}")
        if not hasattr(self.model, "deploy_actor") or not hasattr(self.model, "critic"):
            raise ValueError(f"Trainable scope {scope!r} requires the attention policy architecture.")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in (self.model.deploy_actor, self.model.critic):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if bool(self.config.freeze_action_std):
            self.model.log_std.requires_grad_(False)
        self.configure_optimizer()

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.distribution(obs)
            action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action.squeeze(0), -self.config.action_clip, self.config.action_clip).cpu().numpy()

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "config": self.config.__dict__,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        sidecar = {
            "algorithm": "ppo",
            "weights": path.name,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": self.config.__dict__,
        }
        if metadata:
            sidecar["metadata"] = metadata
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def train_ppo(
    env_factory: Callable[[int], object],
    observation_dim: int,
    action_dim: int,
    config: PPOConfig,
    progress_every: int = 5,
    agent: PPOAgent | None = None,
    teacher_model: nn.Module | None = None,
) -> tuple[PPOAgent, pd.DataFrame]:
    if agent is None:
        agent = PPOAgent(observation_dim, action_dim, config)
    history = []
    completed_episodes = 0
    update_index = 0

    while completed_episodes < config.episodes:
        rollout = _collect_rollout(env_factory, agent, config, completed_episodes)
        completed_episodes += rollout["episode_count"]
        update_index += 1
        teacher_coef = _teacher_kl_coef_for_update(config, update_index)
        losses = _ppo_update(agent, rollout, config, teacher_model=teacher_model, teacher_kl_coef=teacher_coef)
        dual_stats = _update_dual(config, rollout)
        losses.update(dual_stats)
        for row in rollout["episode_rows"]:
            row.update(losses)
        history.extend(rollout["episode_rows"])

        if progress_every and update_index % progress_every == 0:
            recent = history[-max(1, config.rollout_episodes) :]
            reward = sum(row["total_reward"] for row in recent) / len(recent)
            profit = sum(row["cumulative_profit"] for row in recent) / len(recent)
            breach = sum(row["liquidity_breach"] for row in recent) / len(recent)
            print(
                f"update={update_index} episodes={completed_episodes}/{config.episodes} "
                f"reward={reward:.6f} profit={profit:.2f} breach={breach:.3f} "
                f"dual={losses.get('dual_lambda', 0.0):.4f} "
                f"policy_loss={losses['policy_loss']:.6f} value_loss={losses['value_loss']:.6f} "
                f"teacher_kl={losses.get('teacher_kl_loss', 0.0):.6f}",
                flush=True,
            )
        _maybe_save_training_checkpoint(agent, config, update_index, completed_episodes)

    return agent, pd.DataFrame(history)


def train_ppo_torch_batch(
    env_batch_factory: Callable[[int, int], object],
    observation_dim: int,
    action_dim: int,
    config: PPOConfig,
    progress_every: int = 5,
    agent: PPOAgent | None = None,
    teacher_model: nn.Module | None = None,
    gate_callback: Callable[[PPOAgent, int, int], dict[str, float | bool]] | None = None,
) -> tuple[PPOAgent, pd.DataFrame]:
    if agent is None:
        agent = PPOAgent(observation_dim, action_dim, config)
    history = []
    completed_episodes = 0
    update_index = 0

    while completed_episodes < config.episodes:
        batch_size = min(config.rollout_episodes, config.episodes - completed_episodes)
        env = env_batch_factory(completed_episodes, batch_size)
        rollout = _collect_torch_batch_rollout(env, agent, config, completed_episodes)
        completed_episodes += rollout["episode_count"]
        update_index += 1
        previous_model_state = None
        previous_optimizer_state = None
        if gate_callback is not None:
            previous_model_state = copy.deepcopy(agent.model.state_dict())
            previous_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())
        teacher_coef = _teacher_kl_coef_for_update(config, update_index)
        losses = _ppo_update(agent, rollout, config, teacher_model=teacher_model, teacher_kl_coef=teacher_coef)
        dual_stats = _update_dual(config, rollout)
        losses.update(dual_stats)
        if gate_callback is not None:
            gate_stats = gate_callback(agent, update_index, completed_episodes)
            if not bool(gate_stats.get("gate_accepted", False)):
                assert previous_model_state is not None
                assert previous_optimizer_state is not None
                candidate_model_state = copy.deepcopy(agent.model.state_dict())
                for blend_factor in getattr(config, "gate_blend_factors", None) or []:
                    blend_factor = float(blend_factor)
                    if blend_factor <= 0.0 or blend_factor >= 1.0:
                        continue
                    agent.model.load_state_dict(
                        _blend_state_dict(previous_model_state, candidate_model_state, blend_factor)
                    )
                    blended_stats = gate_callback(agent, update_index, completed_episodes)
                    blended_stats["gate_blend_factor"] = blend_factor
                    if bool(blended_stats.get("gate_accepted", False)):
                        agent.optimizer.load_state_dict(previous_optimizer_state)
                        gate_stats = blended_stats
                        break
                if not bool(gate_stats.get("gate_accepted", False)):
                    agent.model.load_state_dict(previous_model_state)
                    agent.optimizer.load_state_dict(previous_optimizer_state)
            losses.update({key: float(value) for key, value in gate_stats.items()})
        for row in rollout["episode_rows"]:
            row.update(losses)
        history.extend(rollout["episode_rows"])

        if progress_every and update_index % progress_every == 0:
            recent = history[-max(1, config.rollout_episodes) :]
            reward = sum(row["total_reward"] for row in recent) / len(recent)
            profit = sum(row["cumulative_profit"] for row in recent) / len(recent)
            breach = sum(row["liquidity_breach"] for row in recent) / len(recent)
            print(
                f"update={update_index} episodes={completed_episodes}/{config.episodes} "
                f"reward={reward:.6f} profit={profit:.2f} breach={breach:.3f} "
                f"dual={losses.get('dual_lambda', 0.0):.4f} "
                f"policy_loss={losses['policy_loss']:.6f} value_loss={losses['value_loss']:.6f} "
                f"teacher_kl={losses.get('teacher_kl_loss', 0.0):.6f}",
                flush=True,
            )
        _maybe_save_training_checkpoint(agent, config, update_index, completed_episodes)

    return agent, pd.DataFrame(history)


def _collect_torch_batch_rollout(env, agent: PPOAgent, config: PPOConfig, episode_offset: int) -> dict:
    obs = env.reset()
    episode_buffers = [
        {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "constraint_costs": [],
            "dones": [],
            "values": [],
            "next_values": [],
        }
        for _ in range(env.batch_size)
    ]

    while not bool(env.done.all().item()):
        active = ~env.done
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        active_obs = obs[active]
        with torch.no_grad():
            dist = agent.model.distribution(active_obs)
            action_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)
            value_tensor = agent.model.value(active_obs)
            env_action = torch.clamp(action_tensor, -config.action_clip, config.action_clip)

        full_action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=agent.device)
        full_action[active] = env_action
        next_obs, reward, done, info = env.step(full_action)

        done_active = done[active]
        reward_active = reward[active]
        constraint_cost_active = _torch_constraint_cost(info, config, env.config.initial_cash)[active]
        active_next = ~done_active
        next_value_tensor = torch.zeros(active_obs.shape[0], dtype=torch.float32, device=agent.device)
        if bool(active_next.any().item()):
            with torch.no_grad():
                next_value_tensor[active_next] = agent.model.value(next_obs[active][active_next])

        for row, env_index in enumerate(active_indices.tolist()):
            buffer = episode_buffers[int(env_index)]
            buffer["observations"].append(active_obs[row])
            buffer["actions"].append(action_tensor[row])
            buffer["log_probs"].append(log_prob_tensor[row])
            buffer["rewards"].append(reward_active[row])
            buffer["constraint_costs"].append(constraint_cost_active[row])
            buffer["dones"].append(done_active[row].float())
            buffer["values"].append(value_tensor[row])
            buffer["next_values"].append(next_value_tensor[row])
        obs = next_obs

    observations = []
    actions = []
    log_probs = []
    advantages = []
    returns = []
    values = []
    for buffer in episode_buffers:
        obs_tensor = torch.stack(buffer["observations"])
        action_buffer = torch.stack(buffer["actions"])
        log_prob_buffer = torch.stack(buffer["log_probs"])
        reward_buffer = torch.stack(buffer["rewards"])
        constraint_cost_buffer = torch.stack(buffer["constraint_costs"])
        adjusted_reward_buffer = reward_buffer - float(config.dual_lambda) * constraint_cost_buffer
        done_buffer = torch.stack(buffer["dones"])
        value_buffer = torch.stack(buffer["values"])
        next_value_buffer = torch.stack(buffer["next_values"])
        advantage_buffer, return_buffer = _compute_gae_torch(
            adjusted_reward_buffer,
            done_buffer,
            value_buffer,
            next_value_buffer,
            config.gamma,
            config.gae_lambda,
        )
        observations.append(obs_tensor)
        actions.append(action_buffer)
        log_probs.append(log_prob_buffer)
        advantages.append(advantage_buffer)
        returns.append(return_buffer)
        values.append(value_buffer)

    frame = env.episode_frame()
    episode_rows = []
    for i, row in frame.iterrows():
        episode_rows.append(
            {
                "episode": episode_offset + int(i) + 1,
                "train_scenario": env.scenario_names[int(i)],
                "calibration_name": env.calibrations[int(i)].name,
                "starting_cash": env.starting_cash_values[int(i)],
                "total_reward": float(row.get("total_reward", 0.0)),
                "cumulative_profit": float(row.get("period_profit", 0.0)),
                "decision_period_profit": float(row.get("decision_period_profit", 0.0)),
                "terminal_runoff_profit": float(row.get("terminal_runoff_profit", 0.0)),
                "terminal_value": float(row.get("terminal_value", 0.0)),
                "terminal_budget": float(row.get("terminal_budget", 0.0)),
                "ending_exposure": float(row.get("ending_exposure", 0.0)),
                "liquidity_breach": float(row.get("liquidity_breach", 0.0)),
                "liquidity_breach_count": float(row.get("liquidity_breach_count", 0.0)),
                "liquidated": float(row.get("liquidated", 0.0)),
                "constraint_cost": _episode_constraint_cost(row, config),
                "constraint_violation": _episode_constraint_violation(row, config),
                "liquidation_loss": float(row.get("liquidation_loss", 0.0)),
                "disbursement": float(row.get("disbursement", 0.0)),
                "default_loss": float(row.get("default_loss", 0.0)),
                "lcr_buffer_violation": float(row.get("lcr_buffer_violation", 0.0)),
                "car_buffer_violation": float(row.get("car_buffer_violation", 0.0)),
                "soft_risk_penalty": float(row.get("soft_risk_penalty", 0.0)),
            }
        )

    return {
        "observations": torch.cat(observations),
        "actions": torch.cat(actions),
        "old_log_probs": torch.cat(log_probs),
        "advantages": torch.cat(advantages),
        "returns": torch.cat(returns),
        "values": torch.cat(values),
        "episode_rows": episode_rows,
        "episode_count": env.batch_size,
    }


def _collect_rollout(env_factory, agent: PPOAgent, config: PPOConfig, episode_offset: int) -> dict:
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    next_values = []
    episode_rows = []
    episode_count = min(config.rollout_episodes, config.episodes - episode_offset)

    envs = []
    current_obs: list[np.ndarray | None] = []
    active = np.ones(episode_count, dtype=bool)
    episode_buffers = [
        {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "constraint_costs": [],
            "dones": [],
            "values": [],
            "next_values": [],
            "total_reward": 0.0,
        }
        for _ in range(episode_count)
    ]

    for local_episode in range(episode_count):
        seed = config.seed + episode_offset + local_episode
        env = env_factory(seed)
        obs, _ = env.reset(seed=seed)
        envs.append(env)
        current_obs.append(obs)

    while np.any(active):
        active_indices = np.flatnonzero(active)
        obs_batch_np = np.stack([current_obs[index] for index in active_indices]).astype(np.float32, copy=False)
        obs_tensor = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            dist = agent.model.distribution(obs_tensor)
            action_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)
            value_tensor = agent.model.value(obs_tensor)
            env_action_tensor = torch.clamp(action_tensor, -config.action_clip, config.action_clip)

        raw_actions = action_tensor.detach().cpu().numpy()
        env_actions = env_action_tensor.detach().cpu().numpy()
        batch_log_probs = log_prob_tensor.detach().cpu().numpy()
        batch_values = value_tensor.detach().cpu().numpy()

        stepped: list[tuple[int, np.ndarray, float, float, bool]] = []
        pending_next_obs = []
        pending_positions = []
        for batch_row, env_index in enumerate(active_indices):
            env = envs[env_index]
            next_obs, reward, terminated, truncated, info = env.step(env_actions[batch_row])
            done = bool(terminated or truncated)
            constraint_cost = _numpy_constraint_cost(info, config, env.initial_cash)
            stepped.append((env_index, next_obs, float(reward), float(constraint_cost), done))
            if not done:
                pending_positions.append(batch_row)
                pending_next_obs.append(next_obs)

        batch_next_values = np.zeros(len(active_indices), dtype=np.float32)
        if pending_next_obs:
            next_batch_np = np.stack(pending_next_obs).astype(np.float32, copy=False)
            next_tensor = torch.as_tensor(next_batch_np, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                next_value_tensor = agent.model.value(next_tensor)
            batch_next_values[np.asarray(pending_positions, dtype=np.int64)] = (
                next_value_tensor.detach().cpu().numpy()
            )

        for batch_row, (env_index, next_obs, reward, constraint_cost, done) in enumerate(stepped):
            buffer = episode_buffers[env_index]
            buffer["observations"].append(current_obs[env_index])
            buffer["actions"].append(raw_actions[batch_row])
            buffer["log_probs"].append(float(batch_log_probs[batch_row]))
            buffer["rewards"].append(float(reward))
            buffer["constraint_costs"].append(float(constraint_cost))
            buffer["dones"].append(float(done))
            buffer["values"].append(float(batch_values[batch_row]))
            buffer["next_values"].append(float(batch_next_values[batch_row]))
            buffer["total_reward"] += float(reward)
            if done:
                active[env_index] = False
                current_obs[env_index] = None
            else:
                current_obs[env_index] = next_obs

    for local_episode, buffer in enumerate(episode_buffers):
        observations.extend(buffer["observations"])
        actions.extend(buffer["actions"])
        log_probs.extend(buffer["log_probs"])
        rewards.extend(buffer["rewards"])
        dones.extend(buffer["dones"])
        values.extend(buffer["values"])
        next_values.extend(buffer["next_values"])
        episode_rows.append(
            _training_episode_row(
                envs[local_episode],
                total_reward=float(buffer["total_reward"]),
                episode_number=episode_offset + local_episode + 1,
                config=config,
            )
        )

    per_episode_advantages = []
    per_episode_returns = []
    for buffer in episode_buffers:
        episode_rewards = [
            float(reward) - float(config.dual_lambda) * float(cost)
            for reward, cost in zip(buffer["rewards"], buffer["constraint_costs"])
        ]
        advantage_buffer, return_buffer = _compute_gae(
            episode_rewards,
            buffer["dones"],
            buffer["values"],
            buffer["next_values"],
            config.gamma,
            config.gae_lambda,
        )
        per_episode_advantages.extend(advantage_buffer.tolist())
        per_episode_returns.extend(return_buffer.tolist())
    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=agent.device),
        "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=agent.device),
        "old_log_probs": torch.as_tensor(log_probs, dtype=torch.float32, device=agent.device),
        "advantages": torch.as_tensor(per_episode_advantages, dtype=torch.float32, device=agent.device),
        "returns": torch.as_tensor(per_episode_returns, dtype=torch.float32, device=agent.device),
        "values": torch.as_tensor(values, dtype=torch.float32, device=agent.device),
        "episode_rows": episode_rows,
        "episode_count": episode_count,
    }


def _training_episode_row(env, total_reward: float, episode_number: int, config: PPOConfig | None = None) -> dict[str, float | str]:
    trace = env.trace
    breach_count = sum(int(row.get("liquidity_breach_count", int(row.get("liquidity_breach", False)))) for row in trace)
    cumulative_profit = sum(float(row.get("period_profit", 0.0)) for row in trace)
    terminal_runoff_profit = sum(float(row.get("terminal_runoff_profit", 0.0)) for row in trace)
    terminal_value = sum(float(row.get("terminal_value", 0.0)) for row in trace)
    decision_period_profit = sum(float(row.get("decision_period_profit", 0.0)) for row in trace)
    disbursement = sum(float(row.get("disbursement", 0.0)) for row in trace)
    default_loss = sum(float(row.get("default_loss", 0.0)) for row in trace)
    liquidation_loss = sum(float(row.get("liquidation_loss", 0.0)) for row in trace)
    lcr_buffer_violation = max([float(row.get("lcr_buffer_violation", 0.0)) for row in trace] or [0.0])
    car_buffer_violation = max([float(row.get("car_buffer_violation", 0.0)) for row in trace] or [0.0])
    soft_risk_penalty = sum(float(row.get("soft_risk_penalty", 0.0)) for row in trace)
    expected_shortfall = max([float(row.get("max_liquidity_shortfall", 0.0)) for row in trace] or [0.0])
    liquidated = any(bool(row.get("liquidated", False)) for row in trace)
    output = {
        "episode": episode_number,
        "train_scenario": getattr(env, "scenario_name", getattr(env.calibration, "name", "")),
        "calibration_name": getattr(env.calibration, "name", ""),
        "starting_cash": float(getattr(env, "starting_cash", env.initial_cash)),
        "total_reward": total_reward,
        "cumulative_profit": cumulative_profit,
        "decision_period_profit": decision_period_profit,
        "terminal_runoff_profit": terminal_runoff_profit,
        "terminal_value": terminal_value,
        "terminal_budget": float(env.budget),
        "ending_exposure": float(env.loan_book.sum()),
        "liquidity_breach": float(breach_count > 0),
        "liquidity_breach_count": breach_count,
        "liquidated": float(liquidated),
        "expected_shortfall": expected_shortfall,
        "expected_shortfall_ratio": expected_shortfall / max(float(env.initial_cash), 1.0),
        "liquidation_loss": liquidation_loss,
        "disbursement": disbursement,
        "default_loss": default_loss,
        "lcr_buffer_violation": lcr_buffer_violation,
        "car_buffer_violation": car_buffer_violation,
        "soft_risk_penalty": soft_risk_penalty,
    }
    if config is not None:
        output["constraint_cost"] = _episode_constraint_cost(output, config)
        output["constraint_violation"] = _episode_constraint_violation(output, config)
    return output


def _torch_constraint_cost(info: dict, config: PPOConfig, initial_cash: float) -> torch.Tensor:
    if not config.primal_dual:
        sample = next(iter(info.values()))
        return torch.zeros_like(sample, dtype=torch.float32)
    metric = config.constraint_metric
    if metric == "liquidated":
        return info.get("liquidated", torch.zeros_like(next(iter(info.values())))).float()
    if metric == "liquidity_breach":
        return info.get("liquidity_breach", torch.zeros_like(next(iter(info.values())))).float()
    if metric == "expected_shortfall":
        shortfall = info.get("liquidity_shortfall", torch.zeros_like(next(iter(info.values())))).float()
        return shortfall / max(float(initial_cash), 1.0)
    if metric in {"lcr_buffer_violation", "car_buffer_violation", "soft_risk_penalty"}:
        return info.get(metric, torch.zeros_like(next(iter(info.values())))).float()
    raise ValueError(f"Unknown constraint_metric: {metric}")


def _numpy_constraint_cost(info: dict, config: PPOConfig, initial_cash: float) -> float:
    if not config.primal_dual:
        return 0.0
    metric = config.constraint_metric
    if metric == "liquidated":
        return float(bool(info.get("liquidated", False)))
    if metric == "liquidity_breach":
        return float(bool(info.get("liquidity_breach", False)))
    if metric == "expected_shortfall":
        return float(info.get("liquidity_shortfall", 0.0)) / max(float(initial_cash), 1.0)
    if metric in {"lcr_buffer_violation", "car_buffer_violation", "soft_risk_penalty"}:
        return float(info.get(metric, 0.0))
    raise ValueError(f"Unknown constraint_metric: {metric}")


def _episode_constraint_cost(row, config: PPOConfig) -> float:
    metric = config.constraint_metric
    if metric == "liquidated":
        return float(row.get("liquidated", 0.0))
    if metric == "liquidity_breach":
        return float(row.get("liquidity_breach", 0.0))
    if metric == "expected_shortfall":
        return float(row.get("expected_shortfall_ratio", row.get("expected_shortfall", 0.0)))
    if metric in {"lcr_buffer_violation", "car_buffer_violation", "soft_risk_penalty"}:
        return float(row.get(metric, 0.0))
    raise ValueError(f"Unknown constraint_metric: {metric}")


def _episode_constraint_violation(row, config: PPOConfig) -> float:
    return _episode_constraint_cost(row, config) - float(config.constraint_threshold)


def _update_dual(config: PPOConfig, rollout: dict) -> dict[str, float]:
    if not config.primal_dual:
        return {
            "dual_lambda": 0.0,
            "constraint_cost_mean": 0.0,
            "constraint_violation_mean": 0.0,
        }
    violations = [float(row.get("constraint_violation", 0.0)) for row in rollout.get("episode_rows", [])]
    costs = [float(row.get("constraint_cost", 0.0)) for row in rollout.get("episode_rows", [])]
    mean_violation = float(np.mean(violations)) if violations else 0.0
    mean_cost = float(np.mean(costs)) if costs else 0.0
    config.dual_lambda = float(
        np.clip(
            float(config.dual_lambda) + float(config.dual_learning_rate) * mean_violation,
            0.0,
            float(config.dual_max_lambda),
        )
    )
    return {
        "dual_lambda": float(config.dual_lambda),
        "constraint_cost_mean": mean_cost,
        "constraint_violation_mean": mean_violation,
    }


def _teacher_kl_coef_for_update(config: PPOConfig, update_index: int) -> float:
    base = max(float(config.teacher_kl_coef), 0.0)
    if base <= 0.0:
        return 0.0
    stop_update = int(getattr(config, "teacher_kl_stop_update", 0) or 0)
    if stop_update > 0 and int(update_index) > stop_update:
        return 0.0
    decay = float(config.teacher_kl_decay)
    if decay < 0.0:
        decay = 0.0
    coef = base * (decay ** max(int(update_index) - 1, 0))
    return max(float(config.teacher_kl_min_coef), float(coef))


def _maybe_save_training_checkpoint(agent: PPOAgent, config: PPOConfig, update_index: int, completed_episodes: int) -> None:
    interval = int(getattr(config, "checkpoint_every_updates", 0) or 0)
    checkpoint_dir = getattr(config, "checkpoint_dir", None)
    if interval <= 0 or not checkpoint_dir or int(update_index) % interval != 0:
        return
    path = Path(checkpoint_dir) / f"segment_ppo_update_{int(update_index):04d}_episodes_{int(completed_episodes)}.pt"
    agent.save(
        path,
        metadata={
            "methodology": "intermediate_ppo_training_checkpoint",
            "update_index": int(update_index),
            "completed_episodes": int(completed_episodes),
        },
    )


def _blend_state_dict(previous: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], factor: float) -> dict[str, torch.Tensor]:
    factor = float(factor)
    blended = {}
    for key, previous_value in previous.items():
        candidate_value = candidate[key]
        if torch.is_floating_point(previous_value):
            blended[key] = previous_value * (1.0 - factor) + candidate_value * factor
        else:
            blended[key] = candidate_value
    return blended


def _ppo_update(
    agent: PPOAgent,
    rollout: dict,
    config: PPOConfig,
    teacher_model: nn.Module | None = None,
    teacher_kl_coef: float = 0.0,
) -> dict[str, float]:
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = observations.shape[0]
    minibatch_size = min(config.minibatch_size, sample_count)
    teacher_active = teacher_model is not None and float(teacher_kl_coef) > 0.0
    losses = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "teacher_kl_loss": 0.0,
        "teacher_kl_coef": float(teacher_kl_coef) if teacher_active else 0.0,
    }
    update_count = 0

    for _ in range(config.update_epochs):
        indices = torch.randperm(sample_count, device=agent.device)
        for start in range(0, sample_count, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            dist = agent.model.distribution(observations[batch_idx])
            new_log_probs = dist.log_prob(actions[batch_idx]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            new_values = agent.model.value(observations[batch_idx])
            ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * torch.mean((returns[batch_idx] - new_values) ** 2)
            teacher_kl_loss = torch.zeros((), dtype=torch.float32, device=agent.device)
            if teacher_active:
                with torch.no_grad():
                    teacher_dist = teacher_model.distribution(observations[batch_idx])
                teacher_kl_loss = kl_divergence(dist, teacher_dist).sum(dim=-1).mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + float(teacher_kl_coef) * teacher_kl_loss
            )

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), config.max_grad_norm)
            agent.optimizer.step()

            losses["policy_loss"] += float(policy_loss.detach().cpu().item())
            losses["value_loss"] += float(value_loss.detach().cpu().item())
            losses["entropy"] += float(entropy.detach().cpu().item())
            losses["teacher_kl_loss"] += float(teacher_kl_loss.detach().cpu().item())
            update_count += 1

    averaged = {}
    for key, value in losses.items():
        if key == "teacher_kl_coef":
            averaged[key] = float(value)
        else:
            averaged[key] = value / max(update_count, 1)
    return averaged


def _compute_gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    next_values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _compute_gae_torch(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
