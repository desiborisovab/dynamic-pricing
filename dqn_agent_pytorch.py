"""
Deep Q-Network (DQN) — PyTorch implementation
Architecture: FC(state_dim) → 30 → 30 → 30 → 30 → n_actions
Drop-in replacement for dqn_agent.py
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, n_actions),
        )
        # He initialisation
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# DQN Agent
class DQNAgent:
    def __init__(
            self,
            state_dim: int,
            n_actions: int,
            lr: float = 1e-3,
            gamma: float = 0.95,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            eps_decay: float = 0.995,
            batch_size: int = 64,
            buffer_size: int = 10_000,
            target_update_freq: int = 200,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_freq = target_update_freq
        self.steps = 0
        self.losses = []

        self.online = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)

        print(f" DQNAgent on {DEVICE}")
        print(f" {sum(p.numel() for p in self.online.parameters()):,} parameters")

    # Action selection
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    # Learning step
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, done = self.buffer.sample(self.batch_size)

        # Current Q(s, a)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online to SELECT action, target to EVALUATE it
        with torch.no_grad():
            best_actions = self.online(s2).argmax(dim=1)
            q_next = self.target(s2).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.steps += 1

        # Sync target network
        if self.steps % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss_val

    # Save / Load
    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eps": self.eps,
            "steps": self.steps,
        }, path)
        print(f"  Agent saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps = ckpt["eps"]
        self.steps = ckpt["steps"]
        print(f" Agent loaded ← {path}")
