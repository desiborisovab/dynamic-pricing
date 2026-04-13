# Dynamic Pricing — Deep Q-Network (PyTorch)

An intelligent dynamic pricing system that uses Deep Reinforcement Learning to maximise retail profit by adapting prices to competitor pricing, demand signals, inventory levels, seasonality, and promotions.

**Best result so far: +1120% improvement over fixed baseline pricing** ($938k vs −$92k on held-out sample, 40 episodes × 10k rows, Random Forest trained on raw features).

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

---

## Quickstart

1. Place `retail_store_inventory.csv` inside a `data/` folder
2. Run training:

```
python train.py
```

**Outputs after training:**
- `results.json` — full training history, episode profits, action distributions, eval records
- `dqn_checkpoint.pt` — saved PyTorch model weights (reloadable)
- `sim_scaler.pkl` — fitted StandardScaler (required for production inference)
- `label_encoders.pkl` — fitted LabelEncoders for all categorical features (required for production inference)

---

## Files

| File | Description |
|------|-------------|
| `data_prep.py` | Loads CSV, cleans data, engineers 14 features, clusters products by category + price tier, trains Random Forest demand simulator, saves `sim_scaler.pkl` and `label_encoders.pkl` |
| `environment.py` | RL pricing environment — 10 price actions (×1.00–×1.225), computes profit reward via simulator |
| `dqn_agent_pytorch.py` | PyTorch DQN agent — QNetwork (4×30), ReplayBuffer, Double DQN, Adam, gradient clipping, save/load |
| `dqn_agent.py` | NumPy DQN fallback — identical logic, no PyTorch required |
| `train.py` | Main entry point — baseline calculation, training loop, greedy eval, saves results |
| `predict_price.py` | Production inference — loads checkpoint and returns a price recommendation for any live product |
| `dynamic_pricing_dashboard.html` | Interactive results dashboard — open in any browser |
| `requirements.txt` | Python dependencies |
| `results.json` | Generated after training — feeds the dashboard |
| `dqn_checkpoint.pt` | Generated after training — saved model weights |
| `sim_scaler.pkl` | Generated after training — StandardScaler fitted on training features |
| `label_encoders.pkl` | Generated after training — LabelEncoder mappings for category, region, weather, season |

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
| `EPISODES` | 40 | Training episodes — agent still improving at 40, try 80 |
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

```python
from dqn_agent_pytorch import DQNAgent   # PyTorch (default, recommended)
# from dqn_agent import DQNAgent         # NumPy fallback — no install needed
```

---

## Demand Simulator

The RL agent trains inside a **Random Forest** demand simulator (not the real world).

**Critical:** the Random Forest must be trained on **raw (unscaled) features** — trees use threshold comparisons, not feature magnitudes, so scaling adds no benefit and can hurt performance. The `StandardScaler` is still fitted and saved separately for use by the neural network's state input.

```
# In data_prep.py
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # scaler saved for the RL state vector
X_scaled = np.clip(X_scaled, -10, 10)

model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X, y)   # ← raw features, not X_scaled
```

| Simulator | R² | Notes |
|-----------|-----|-------|
| Linear Regression | 0.348 | Fast, interpretable, poor at non-linear interactions |
| Random Forest (scaled input) | ~0.371 | Underperforms — do not scale before Random Forest |
| Random Forest (raw input) | ~0.75+ | **Current default** — correct configuration |
| XGBoost | ~0.80+ | Best accuracy, requires `brew install libomp` on Mac |

---

## Production Inference

After training, load the checkpoint and saved artefacts to price any product in real time.

**Step 1 — get your exact label encoder mappings** (run once after training):

```
python3 -c "
import pickle
with open('label_encoders.pkl', 'rb') as f:
    enc = pickle.load(f)
for k, v in enc.items():
    print(k, v)
"
```

**Step 2 — call `get_price`:**

```python
from predict_price import get_price

result = get_price({
    "current_price": 33.50,
    "competitor_price": 29.69,
    "inventory_level": 231,
    "discount": 20,
    "holiday_promotion": 0,
    "date": "2026-03-08",
    "category": "Toys",      # must match exact strings from your dataset
    "region": "North",
    "weather": "Rainy",
    "season": "Spring"
})
# → {"recommended_price": 21.61, "effective_price": 17.29, "action": 3, "multiplier": 1.075}
```

> **Important:** `sim_scaler.pkl` and `label_encoders.pkl` must come from the same training run as `dqn_checkpoint.pt`. Mixing artefacts from different runs will produce wrong prices.

---

## Results

| Run | Episodes | Sample Size | Simulator | Loss | Eval Profit | Improvement |
|-----|----------|-------------|-----------|------|-------------|-------------|
| v1 (NumPy) | 8 | 3,000 | Linear Regression | ~37 | $452k | +416% |
| v2 (PyTorch) | 8 | 3,000 | Linear Regression | ~0.37 | $1.1M | +872% |
| v3 (PyTorch) | 20 | 10,000 | Random Forest (scaled) | ~0.33 | $4.4M | +1051% |
| v4 (PyTorch) | 40 | 10,000 | **Random Forest (raw)** | **~0.01** | **$938k** | **+1120%** |

> Note: v4 absolute profit is lower than v3 because the corrected simulator gives more realistic (conservative) demand estimates. The v3 numbers were inflated by the noisy scaled-input simulator. v4 loss of 0.01 vs 0.33 confirms the agent is learning a much tighter, more accurate Q-function.

---
