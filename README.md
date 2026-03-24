# Dynamic Pricing — Deep Q-Network (PyTorch)

An intelligent dynamic pricing system that uses Deep Reinforcement Learning to maximise retail profit by adapting prices to competitor pricing, demand signals, inventory levels, seasonality, and promotions.

**Best result so far: +1050% improvement over fixed baseline pricing** ($4.4M vs −$463k on held-out sample, 20 episodes × 10k rows, Random Forest simulator).

---

## How It Works

1. A **Random Forest** is trained on historical data to simulate demand (units sold) given any product context and price
2. A **Deep Q-Network (DQN) agent** trains inside that simulator — trying 10 different price multipliers and learning which generates the most profit in each context
3. After training, the agent's checkpoint can be loaded to generate real-time price recommendations

```
CSV data → data_prep.py → Random Forest simulator
                                    ↕
                         environment.py ↔ train.py ↔ dqn_agent_pytorch.py
                                    ↓
                              results.json → dashboard.html
                              dqn_checkpoint.pt → predict_price.py (production)
```

---

## Setup

```
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from https://pytorch.org/get-started/locally/

> **Mac users:** If you see an `xgboost` OpenMP error, run `brew install libomp`. If you don't have Homebrew, use the Random Forest simulator instead (already the default).

---

## Quickstart

1. Place `retail_store_inventory.csv` in the project folder
2. Run training:

```
python train.py
```

3. Open `dynamic_pricing_dashboard.html` in a browser to see results

**Outputs after training:**
- `results.json` — full training history, episode profits, action distributions, eval records
- `dqn_checkpoint.pt` — saved PyTorch model weights (reloadable)
- `sim_scaler.pkl` — fitted StandardScaler (required for production inference)

---

## Files

| File | Description |
|------|-------------|
| `data_prep.py` | Loads CSV, cleans data, engineers 14 features, clusters products by category + price tier, trains Random Forest demand simulator |
| `environment.py` | RL pricing environment — 10 price actions (×1.00–×1.225), computes profit reward via simulator |
| `dqn_agent_pytorch.py` | PyTorch DQN agent — QNetwork (4×30), ReplayBuffer, Double DQN, Adam, gradient clipping, save/load |
| `dqn_agent.py` | NumPy DQN fallback — identical logic, no PyTorch required |
| `train.py` | Main entry point — baseline calculation, training loop, greedy eval, saves results |
| `predict_price.py` | Production inference — load checkpoint and get a price recommendation for any live product |
| `dynamic_pricing_dashboard.html` | Interactive results dashboard — open in any browser |
| `requirements.txt` | Python dependencies |
| `results.json` | Generated after training — feeds the dashboard |
| `dqn_checkpoint.pt` | Generated after training — saved model weights |

---

## Neural Network Architecture

```
Input: 14 features (price, competitor price, inventory, discount,
       holiday, day_of_week, month, quarter, price_ratio,
       effective_price, category, region, weather, season)

  → Linear(14 → 30) + ReLU
  → Linear(30 → 30) + ReLU
  → Linear(30 → 30) + ReLU
  → Linear(30 → 30) + ReLU
  → Linear(30 → 10)

Output: 10 Q-values, one per price action
        Agent picks action with highest Q-value

Price actions: base_cost × [1.000, 1.025, 1.050, 1.075, 1.100,
                             1.125, 1.150, 1.175, 1.200, 1.225]
```

Total parameters: 3,550

---

## Hyperparameters

| Param | Best Run Value | Description |
|-------|---------------|-------------|
| `EPISODES` | 20 | Training episodes (more = better, diminishing returns after ~30) |
| `SAMPLE_SIZE` | 10,000 | Rows sampled per episode from the full dataset |
| `LR` | 1e-4 | Adam learning rate — lower = more stable convergence |
| `GAMMA` | 0.95 | Discount factor — weights future vs immediate reward |
| `EPS_START` | 1.0 | Initial exploration rate (fully random) |
| `EPS_END` | 0.05 | Minimum exploration rate (5% random actions) |
| `EPS_DECAY` | 0.999 | Epsilon multiplied by this each step — slower decay = more exploration |
| `BATCH_SIZE` | 64 | Experiences sampled per gradient update |
| `BUFFER_SIZE` | 50,000 | Replay buffer capacity (~5 episodes of memory) |
| `TARGET_UPDATE` | 500 | Steps between target network syncs |

---

## Switch Between PyTorch / NumPy

In `train.py`, swap the import at the top:

```
from dqn_agent_pytorch import DQNAgent   # PyTorch (default, recommended)
# from dqn_agent import DQNAgent         # NumPy fallback — no install needed
```

---

## Demand Simulator

The RL agent trains inside a **Random Forest** demand simulator (not the real world):

```
# In data_prep.py — swap simulator here
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)

# Or with XGBoost (better R², requires libomp on Mac):
# from xgboost import XGBRegressor
# model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
```

| Simulator | R² | Notes |
|-----------|-----|-------|
| Linear Regression | 0.348 | Fast, interpretable, poor at non-linear interactions |
| Random Forest | ~0.75 | Current default — good balance of accuracy and speed |
| XGBoost | ~0.80+ | Best accuracy, requires `brew install libomp` on Mac |

---

## Production Inference

To price a product in real time using the trained agent:

```
from predict_price import get_price

result = get_price({
    "current_price": 33.50,
    "competitor_price": 29.69,
    "inventory_level": 231,
    "discount": 20,
    "holiday_promotion": 0,
    "date": "2026-03-08",
    "category": "Toys",
    "region": "North",
    "weather": "Rainy",
    "season": "Spring"
})
# → {"recommended_price": 21.61, "effective_price": 17.29, "action": 3, "multiplier": 1.075}
```

> **Important:** Run `train.py` once with `sim_scaler.pkl` saving enabled before using `predict_price.py`. The scaler must match the one used during training.

---

## Results

| Run | Episodes | Sample Size | Simulator | Eval Profit | Improvement |
|-----|----------|-------------|-----------|-------------|-------------|
| v1 (NumPy) | 8 | 3,000 | Linear Regression | $452k | +416% |
| v2 (PyTorch) | 8 | 3,000 | Linear Regression | $1.1M | +872% |
| v3 (PyTorch) | 20 | 10,000 | Random Forest | $4.4M | **+1051%** |

---

## Next Steps

- **Train longer:** `EPISODES = 40` — loss was still declining at episode 20
- **Wider network:** increase hidden layers from 30 → 64 or 128 nodes
- **Add discount to action space:** let the agent also choose the discount level
- **Per-cluster agents:** train separate agents per product category + price tier
- **Deploy as API:** wrap `predict_price.py` in Flask for real-time pricing endpoint
- **Retrain monthly:** roll the training window forward with fresh data to keep the agent current