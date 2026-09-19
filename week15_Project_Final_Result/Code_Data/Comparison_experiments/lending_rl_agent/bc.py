"""Behavioral Cloning models for segment-level lending allocation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# MLP Behavioral Cloning
# ---------------------------------------------------------------------------

class MLPBCModel(nn.Module):
    """MLP that maps state (331-dim) → action (31-dim) via supervised learning."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBehavioralCloning:
    """Train MLP via MSE on (state, action) pairs, then act in simulator."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = MLPBCModel(observation_dim, action_dim, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, states: np.ndarray, actions: np.ndarray) -> list[float]:
        """Supervised training on collected (state, action) pairs."""
        X = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        losses = []
        n = len(X)
        self.model.train()
        for epoch in range(self.epochs):
            idx = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                pred = self.model(X[batch_idx])
                loss = nn.functional.mse_loss(pred, Y[batch_idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            avg = epoch_loss / max(steps, 1)
            losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  MLP BC epoch {epoch+1}/{self.epochs}  loss={avg:.6f}")
        return losses

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return raw action vector given a single observation."""
        self.model.eval()
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.model(obs).squeeze(0).cpu().numpy()
        return action


# ---------------------------------------------------------------------------
# XGBoost Behavioral Cloning
# ---------------------------------------------------------------------------

class XGBBehavioralCloning:
    """XGBoost MultiOutput Regressor that maps state → action."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=seed,
                verbosity=0,
                eval_metric="rmse",
            ),
            n_jobs=-1,
        )

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Supervised training on collected (state, action) pairs."""
        print(f"  XGBoost BC fitting on {len(states):,} samples, action_dim={actions.shape[1]}...")
        self.model.fit(states, actions)
        print("  XGBoost BC fit complete.")

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return raw action vector given a single observation."""
        return self.model.predict(observation.reshape(1, -1))[0].astype(np.float32)
