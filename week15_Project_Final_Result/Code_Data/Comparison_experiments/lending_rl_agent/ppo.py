from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 256) -> None:
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
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.35))

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
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.maturity_bucket_count = int(maturity_bucket_count)
        self.segment_parameter_count = int(segment_parameter_count)
        expected_obs = (
            1
            + self.segment_count
            + self.segment_count * self.maturity_bucket_count
            + self.segment_count * self.segment_parameter_count
        )
        expected_action = self.segment_count + 1
        if observation_dim != expected_obs:
            raise ValueError(f"Attention policy expected observation_dim={expected_obs}, got {observation_dim}")
        if action_dim != expected_action:
            raise ValueError(f"Attention policy expected action_dim={expected_action}, got {action_dim}")

        token_dim = 1 + 1 + self.maturity_bucket_count + self.segment_parameter_count
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
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.deploy_actor[-1].weight)
        nn.init.constant_(self.deploy_actor[-1].bias, -3.0)
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.80))

    def _encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        cursor = 0
        cash = observations[:, cursor : cursor + 1]
        cursor += 1
        demand = observations[:, cursor : cursor + self.segment_count].reshape(batch, self.segment_count, 1)
        cursor += self.segment_count
        exposure = observations[:, cursor : cursor + self.segment_count * self.maturity_bucket_count].reshape(
            batch, self.segment_count, self.maturity_bucket_count
        )
        cursor += self.segment_count * self.maturity_bucket_count
        params = observations[:, cursor : cursor + self.segment_count * self.segment_parameter_count].reshape(
            batch, self.segment_count, self.segment_parameter_count
        )
        cash_token = cash.unsqueeze(1).expand(-1, self.segment_count, -1)
        tokens = torch.cat([cash_token, demand, exposure, params], dim=-1)
        encoded = self.token_projection(tokens) + self.segment_embedding.unsqueeze(0)
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        global_state = torch.cat([pooled, cash], dim=1)
        return encoded, global_state, cash

    def distribution(self, observations: torch.Tensor) -> Normal:
        encoded, global_state, _ = self._encode(observations)
        segment_logits = self.segment_actor(encoded).squeeze(-1)
        deploy_logit = self.deploy_actor(global_state)
        mean = torch.cat([deploy_logit, segment_logits], dim=1)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        _, global_state, _ = self._encode(observations)
        return self.critic(global_state).squeeze(-1)

class SegmentGRUActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        segment_count: int = 30,
        maturity_bucket_count: int = 6,
        segment_parameter_count: int = 4,
        gru_layers: int = 2,
        gru_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.maturity_bucket_count = int(maturity_bucket_count)
        self.segment_parameter_count = int(segment_parameter_count)

        expected_obs = (
            1
            + self.segment_count
            + self.segment_count * self.maturity_bucket_count
            + self.segment_count * self.segment_parameter_count
        )
        expected_action = self.segment_count + 1
        if observation_dim != expected_obs:
            raise ValueError(f"GRU policy expected observation_dim={expected_obs}, got {observation_dim}")
        if action_dim != expected_action:
            raise ValueError(f"GRU policy expected action_dim={expected_action}, got {action_dim}")

        # Attention과 동일한 token 구성
        token_dim = 1 + 1 + self.maturity_bucket_count + self.segment_parameter_count

        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # Transformer 대신 GRU로 30개 segment를 순차 처리
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        # 이하 actor/critic 구조는 Attention과 동일
        self.segment_actor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.deploy_actor = nn.Sequential(
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.deploy_actor[-1].weight)
        nn.init.constant_(self.deploy_actor[-1].bias, -3.0)
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.80))

    def _encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        cursor = 0
        cash = observations[:, cursor : cursor + 1]
        cursor += 1
        demand = observations[:, cursor : cursor + self.segment_count].reshape(batch, self.segment_count, 1)
        cursor += self.segment_count
        exposure = observations[:, cursor : cursor + self.segment_count * self.maturity_bucket_count].reshape(
            batch, self.segment_count, self.maturity_bucket_count
        )
        cursor += self.segment_count * self.maturity_bucket_count
        params = observations[:, cursor : cursor + self.segment_count * self.segment_parameter_count].reshape(
            batch, self.segment_count, self.segment_parameter_count
        )

        # Attention과 동일하게 token 구성
        cash_token = cash.unsqueeze(1).expand(-1, self.segment_count, -1)
        tokens = torch.cat([cash_token, demand, exposure, params], dim=-1)
        projected = self.token_projection(tokens)  # (batch, 30, hidden)

        # Transformer 대신 GRU로 순차 처리
        gru_out, _ = self.gru(projected)             # (batch, 30, hidden)

        # Attention은 mean pooling, GRU는 마지막 hidden state를 global로 사용
        pooled = gru_out[:, -1, :]                   # (batch, hidden)
        global_state = torch.cat([pooled, cash], dim=1)
        return gru_out, global_state, cash

    def distribution(self, observations: torch.Tensor) -> Normal:
        encoded, global_state, _ = self._encode(observations)
        segment_logits = self.segment_actor(encoded).squeeze(-1)
        deploy_logit = self.deploy_actor(global_state)
        mean = torch.cat([deploy_logit, segment_logits], dim=1)
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
    attention_heads: int = 4
    attention_layers: int = 2
    attention_dropout: float = 0.0
    gru_layers: int = 2
    gru_dropout: float = 0.0


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
                attention_heads=config.attention_heads,
                attention_layers=config.attention_layers,
                attention_dropout=config.attention_dropout,
            ).to(self.device)
        elif config.architecture == "gru":
            self.model = SegmentGRUActorCritic(
                observation_dim,
                action_dim,
                hidden_size=config.hidden_size,
                segment_count=config.segment_count,
                maturity_bucket_count=config.maturity_bucket_count,
                segment_parameter_count=config.segment_parameter_count,
                gru_layers=config.gru_layers,
                gru_dropout=config.gru_dropout,
            ).to(self.device)
        elif config.architecture == "mlp":
            self.model = ActorCritic(observation_dim, action_dim, hidden_size=config.hidden_size).to(self.device)
        else:
            raise ValueError(f"Unknown policy architecture: {config.architecture}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

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
) -> tuple[PPOAgent, pd.DataFrame]:
    agent = PPOAgent(observation_dim, action_dim, config)
    history = []
    completed_episodes = 0
    update_index = 0

    while completed_episodes < config.episodes:
        rollout = _collect_rollout(env_factory, agent, config, completed_episodes)
        completed_episodes += rollout["episode_count"]
        update_index += 1
        losses = _ppo_update(agent, rollout, config)
        history.extend(rollout["episode_rows"])

        if progress_every and update_index % progress_every == 0:
            recent = history[-max(1, config.rollout_episodes) :]
            reward = sum(row["total_reward"] for row in recent) / len(recent)
            profit = sum(row["cumulative_profit"] for row in recent) / len(recent)
            breach = sum(row["liquidity_breach"] for row in recent) / len(recent)
            print(
                f"update={update_index} episodes={completed_episodes}/{config.episodes} "
                f"reward={reward:.6f} profit={profit:.2f} breach={breach:.3f} "
                f"policy_loss={losses['policy_loss']:.6f} value_loss={losses['value_loss']:.6f}",
                flush=True,
            )

    return agent, pd.DataFrame(history)


def train_ppo_torch_batch(
    env_batch_factory: Callable[[int, int], object],
    observation_dim: int,
    action_dim: int,
    config: PPOConfig,
    progress_every: int = 5,
) -> tuple[PPOAgent, pd.DataFrame]:
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
        losses = _ppo_update(agent, rollout, config)
        history.extend(rollout["episode_rows"])

        if progress_every and update_index % progress_every == 0:
            recent = history[-max(1, config.rollout_episodes) :]
            reward = sum(row["total_reward"] for row in recent) / len(recent)
            profit = sum(row["cumulative_profit"] for row in recent) / len(recent)
            breach = sum(row["liquidity_breach"] for row in recent) / len(recent)
            print(
                f"update={update_index} episodes={completed_episodes}/{config.episodes} "
                f"reward={reward:.6f} profit={profit:.2f} breach={breach:.3f} "
                f"policy_loss={losses['policy_loss']:.6f} value_loss={losses['value_loss']:.6f}",
                flush=True,
            )

    return agent, pd.DataFrame(history)


def _collect_torch_batch_rollout(env, agent: PPOAgent, config: PPOConfig, episode_offset: int) -> dict:
    obs = env.reset()
    episode_buffers = [
        {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
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
        next_obs, reward, done, _ = env.step(full_action)

        done_active = done[active]
        reward_active = reward[active]
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
        done_buffer = torch.stack(buffer["dones"])
        value_buffer = torch.stack(buffer["values"])
        next_value_buffer = torch.stack(buffer["next_values"])
        advantage_buffer, return_buffer = _compute_gae_torch(
            reward_buffer,
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
                "liquidation_loss": float(row.get("liquidation_loss", 0.0)),
                "disbursement": float(row.get("disbursement", 0.0)),
                "default_loss": float(row.get("default_loss", 0.0)),
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

        stepped: list[tuple[int, np.ndarray, float, bool]] = []
        pending_next_obs = []
        pending_positions = []
        for batch_row, env_index in enumerate(active_indices):
            env = envs[env_index]
            next_obs, reward, terminated, truncated, _ = env.step(env_actions[batch_row])
            done = bool(terminated or truncated)
            stepped.append((env_index, next_obs, float(reward), done))
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

        for batch_row, (env_index, next_obs, reward, done) in enumerate(stepped):
            buffer = episode_buffers[env_index]
            buffer["observations"].append(current_obs[env_index])
            buffer["actions"].append(raw_actions[batch_row])
            buffer["log_probs"].append(float(batch_log_probs[batch_row]))
            buffer["rewards"].append(float(reward))
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
            )
        )

    advantages, returns = _compute_gae(rewards, dones, values, next_values, config.gamma, config.gae_lambda)
    return {
        "observations": torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=agent.device),
        "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=agent.device),
        "old_log_probs": torch.as_tensor(log_probs, dtype=torch.float32, device=agent.device),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=agent.device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=agent.device),
        "values": torch.as_tensor(values, dtype=torch.float32, device=agent.device),
        "episode_rows": episode_rows,
        "episode_count": episode_count,
    }


def _training_episode_row(env, total_reward: float, episode_number: int) -> dict[str, float | str]:
    trace = env.trace
    breach_count = sum(int(row.get("liquidity_breach_count", int(row.get("liquidity_breach", False)))) for row in trace)
    cumulative_profit = sum(float(row.get("period_profit", 0.0)) for row in trace)
    terminal_runoff_profit = sum(float(row.get("terminal_runoff_profit", 0.0)) for row in trace)
    terminal_value = sum(float(row.get("terminal_value", 0.0)) for row in trace)
    decision_period_profit = sum(float(row.get("decision_period_profit", 0.0)) for row in trace)
    disbursement = sum(float(row.get("disbursement", 0.0)) for row in trace)
    default_loss = sum(float(row.get("default_loss", 0.0)) for row in trace)
    liquidation_loss = sum(float(row.get("liquidation_loss", 0.0)) for row in trace)
    liquidated = any(bool(row.get("liquidated", False)) for row in trace)
    return {
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
        "liquidation_loss": liquidation_loss,
        "disbursement": disbursement,
        "default_loss": default_loss,
    }


def _ppo_update(agent: PPOAgent, rollout: dict, config: PPOConfig) -> dict[str, float]:
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = observations.shape[0]
    minibatch_size = min(config.minibatch_size, sample_count)
    losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
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
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), config.max_grad_norm)
            agent.optimizer.step()

            losses["policy_loss"] += float(policy_loss.detach().cpu().item())
            losses["value_loss"] += float(value_loss.detach().cpu().item())
            losses["entropy"] += float(entropy.detach().cpu().item())
            update_count += 1

    return {key: value / max(update_count, 1) for key, value in losses.items()}


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
