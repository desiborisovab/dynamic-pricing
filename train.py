"""
Training loop for the Dynamic Pricing DQN Agent (PyTorch version)
Run: python train.py
"""

import numpy as np
import json
import os

from dqn_agent_pytorch import DQNAgent
# from dqn_agent import DQNAgent   ← NumPy fallback

from data_prep import prepare_all
from environment import PricingEnv, MULTIPLIERS

# Hyperparameters
EPISODES = 40
SAMPLE_SIZE = 10_000  # rows per episode
LR = 1e-4
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
TARGET_UPDATE = 500
SAVE_CHECKPOINT = True  # saves agent weights after training


def train():
    print("=" * 55)
    print("  Dynamic Pricing — Deep Q-Network (PyTorch)")
    print("=" * 55)

    df, sim_model, sim_scaler, feature_cols = prepare_all()
    env = PricingEnv(df, sim_model, sim_scaler, feature_cols, seed=0)

    state_dim = env.n_features
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=env.n_actions,
        lr=LR, gamma=GAMMA,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        target_update_freq=TARGET_UPDATE,
    )

    print(f"\n State dim: {state_dim}")
    print(f"Actions: {env.n_actions} (multipliers: {MULTIPLIERS})")
    print(f"Episodes: {EPISODES} x {SAMPLE_SIZE} rows\n")

    # Baseline
    print("[Baseline] Computing original-pricing profit on sample...")
    sample_df = df.sample(SAMPLE_SIZE, random_state=1)
    env_base = PricingEnv(sample_df, sim_model, sim_scaler, feature_cols, seed=1)
    base_profit = 0.0
    state = env_base.reset()
    for _ in range(SAMPLE_SIZE):
        _, reward, done, info = env_base.step(4)
        base_profit += info["profit"]
        if done: break
    print(f"Baseline total profit: ${base_profit:,.0f}\n")

    history = {
        "episode_reward": [], "episode_profit": [],
        "mean_loss": [], "epsilon": [],
        "baseline_profit": base_profit,
        "action_counts": [],
    }

    for ep in range(1, EPISODES + 1):
        ep_df = df.sample(SAMPLE_SIZE, random_state=ep)
        ep_env = PricingEnv(ep_df, sim_model, sim_scaler, feature_cols, seed=ep)

        state = ep_env.reset()
        ep_reward = 0.0
        ep_profit = 0.0
        ep_losses = []
        action_counts = np.zeros(env.n_actions, dtype=int)

        for _ in range(SAMPLE_SIZE):
            action = agent.act(state, training=True)
            next_state, reward, done, info = ep_env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            ep_profit += info["profit"]
            action_counts[action] += 1
            state = next_state
            if done: break

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["episode_reward"].append(ep_reward)
        history["episode_profit"].append(ep_profit)
        history["mean_loss"].append(mean_loss)
        history["epsilon"].append(agent.eps)
        history["action_counts"].append(action_counts.tolist())

        improvement = (ep_profit - base_profit) / (abs(base_profit) + 1e-9) * 100
        print(f"Episode {ep:2d}/{EPISODES} | "
              f"Profit ${ep_profit:>10,.0f} | "
              f"Δbaseline {improvement:+.2f}% | "
              f"eps {agent.eps:.3f} | "
              f"loss {mean_loss:.4f}")

    print("\n[Eval] Running greedy policy on held-out sample...")
    eval_df = df.sample(SAMPLE_SIZE, random_state=99)
    eval_env = PricingEnv(eval_df, sim_model, sim_scaler, feature_cols, seed=99)
    state = eval_env.reset()
    eval_profit = 0.0
    eval_records = []

    for _ in range(SAMPLE_SIZE):
        action = agent.act(state, training=False)
        next_state, reward, done, info = eval_env.step(action)
        eval_profit += info["profit"]
        eval_records.append({
            "product_id": info["product_id"],
            "date": info["date"],
            "action": int(action),
            "multiplier": float(MULTIPLIERS[action]),
            "price": round(info["price"], 2),
            "demand": round(info["demand"], 1),
            "profit": round(info["profit"], 2),
        })
        state = next_state
        if done: break

    improvement = (eval_profit - base_profit) / (abs(base_profit) + 1e-9) * 100
    print(f"RL agent profit: ${eval_profit:,.0f}")
    print(f"Baseline profit: ${base_profit:,.0f}")
    print(f"Improvement: {improvement:+.2f}%")

    history["eval_profit"] = eval_profit
    history["eval_records"] = eval_records[:500]
    history["final_improvement_pct"] = improvement
    history["multipliers"] = MULTIPLIERS.tolist()

    with open("results.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n Results saved -> results.json")

    if SAVE_CHECKPOINT:
        agent.save("dqn_checkpoint.pt")

    return history


if __name__ == "__main__":
    train()
