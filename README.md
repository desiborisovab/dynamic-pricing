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


pip install -r requirements.txt


For GPU support, install PyTorch with CUDA from https://pytorch.org/get-started/locally/

> **Mac users:** If you see an `xgboost` OpenMP error, run `brew install libomp`. If you don't have Homebrew, use the Random Forest simulator instead (already the default).

---

## Quickstart

1. Place `retail_store_inventory.csv` inside a `data/` folder
2. Run training:

python train.py

3. Open `dynamic_pricing_dashboard.html` in a browser to see results

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


from dqn_agent_pytorch import DQNAgent   # PyTorch (default, recommended)
# from dqn_agent import DQNAgent         # NumPy fallback — no install needed


---

## Demand Simulator

The RL agent trains inside a **Random Forest** demand simulator (not the real world).

**Critical:** the Random Forest must be trained on **raw (unscaled) features** — trees use threshold comparisons, not feature magnitudes, so scaling adds no benefit and can hurt performance. The `StandardScaler` is still fitted and saved separately for use by the neural network's state input.


# In data_prep.py
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # scaler saved for the RL state vector
X_scaled = np.clip(X_scaled, -10, 10)

model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X, y)   # ← raw features, not X_scaled

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


python3 -c "
import pickle
with open('label_encoders.pkl', 'rb') as f:
    enc = pickle.load(f)
for k, v in enc.items():
    print(k, v)
"


**Step 2 — call `get_price`:**


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

## Next Steps

- **Train longer:** `EPISODES = 80` — loss and profit were still trending upward at episode 40
- **Wider network:** increase hidden layers from 30 → 128 nodes for more Q-function capacity
- **Larger buffer:** `BUFFER_SIZE = 100_000` — now that loss is stable at 0.01
- **Add discount to action space:** let the agent also choose the discount level
- **Per-cluster agents:** train separate agents per product category + price tier
- **Deploy as API:** wrap `predict_price.py` in Flask for real-time pricing endpoint
- **Retrain monthly:** roll the training window forward with fresh data to keep the agent current

---

## Azure Deployment

### Architecture

```
Local machine                    Azure
─────────────                    ─────────────────────────────────────
python train.py
       ↓
  dqn_checkpoint.pt  ──────→  Blob Storage (dynamic-pricing-model)
  sim_scaler.pkl                       ↓
  label_encoders.pkl           Container Registry (Docker image)
  results.json                         ↓
                               Container Apps (Flask API)
                                         ↓
                               https://<your-app>.azurecontainerapps.io
```

### Prerequisites


# Install Azure CLI (Mac)
brew install azure-cli

# Install Docker
brew install --cask docker

# Log in to Azure
az login


### First Deployment


# 1. Train the model locally first
python train.py

# 2. Make the deploy script executable
chmod +x azure/deploy.sh

# 3. Edit the config variables at the top of azure/deploy.sh
#    (STORAGE_ACCOUNT and ACR_NAME must be globally unique)

# 4. Deploy everything
./azure/deploy.sh


This will:
- Create a Resource Group, Storage Account, and Blob container
- Upload your model artefacts to Blob Storage
- Build a Docker image and push it to Azure Container Registry
- Deploy the API as an Azure Container App with public HTTPS endpoint

### Redeploy After Retraining


python train.py          # retrain locally
./azure/redeploy.sh      # upload new artefacts + restart API


### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — returns model version and load time |
| `GET` | `/info` | Valid categories, regions, weather, seasons, multipliers |
| `POST` | `/price` | Price recommendation for one product |
| `POST` | `/price/batch` | Price recommendations for up to 500 products |

### Example Request


curl -X POST https://<your-app>.azurecontainerapps.io/price \
  -H "Content-Type: application/json" \
  -d '{
    "product_id":        "P0042",
    "current_price":     33.50,
    "competitor_price":  29.69,
    "inventory_level":   231,
    "discount":          20,
    "holiday_promotion": 0,
    "date":              "2026-04-13",
    "category":          "Toys",
    "region":            "North",
    "weather":           "Rainy",
    "season":            "Spring"
  }'


### Example Response


{
  "product_id":        "P0042",
  "recommended_price": 21.61,
  "effective_price":   17.29,
  "base_cost":         20.10,
  "action":            3,
  "multiplier":        1.075,
  "model_version":     "latest",
  "priced_at":         "2026-04-13T10:23:41.123456"
}


### Test the API


# Against deployed Azure API (reads API_URL from .env)
python azure/test_api.py

# Against local API
API_URL=http://localhost:8000 python azure/test_api.py


### Run API Locally (before deploying)


pip install -r requirements_api.txt
cp .env.template .env          # fill in your connection string
python api/app.py


### Azure Files

| File | Description |
|------|-------------|
| `azure/deploy.sh` | One-shot script — creates all Azure resources and deploys the API |
| `azure/redeploy.sh` | Quick redeploy — uploads new artefacts and restarts the API |
| `azure/upload_artefacts.py` | Uploads model artefacts to Blob Storage after training |
| `azure/test_api.py` | Tests all API endpoints against local or deployed API |
| `api/app.py` | Flask API — `/health`, `/info`, `/price`, `/price/batch` |
| `api/Dockerfile` | Containerises the API |
| `requirements_api.txt` | API-only dependencies (no training libs) |
| `.env.template` | Template for environment variables — copy to `.env` |

### Estimated Azure Costs

| Resource | SKU | Est. Monthly Cost |
|----------|-----|-------------------|
| Storage Account | Standard LRS | ~$0.02/GB |
| Container Registry | Basic | ~$5/month |
| Container Apps | 0.5 CPU / 1GB RAM, 1 replica | ~$10–15/month |
| **Total** | | **~$15–20/month** |