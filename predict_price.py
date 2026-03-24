# predict_price.py
import torch
import numpy as np
from dqn_agent_pytorch import DQNAgent, DEVICE
from environment import MULTIPLIERS

# Load once at startup
STATE_DIM = 14
N_ACTIONS = 10

agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
agent.load("dqn_checkpoint.pt")

# Reuse the same scaler from training
import pickle

with open("sim_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

SEASON_MAP = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}
WEATHER_MAP = {"Cloudy": 0, "Rainy": 1, "Snowy": 2, "Sunny": 3}
REGION_MAP = {"East": 0, "North": 1, "South": 2, "West": 3}
CATEGORY_MAP= {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3, "Toys": 4}


def get_price(product: dict) -> dict:
    """
    product = {
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
    }
    """
    from datetime import datetime
    d = datetime.strptime(product["date"], "%Y-%m-%d")

    price = product["current_price"]
    comp_price = product["competitor_price"]
    discount = product["discount"]

    features = np.array([[
        price,
        comp_price,
        product["inventory_level"],
        discount,
        product["holiday_promotion"],
        d.weekday(),
        d.month,
        (d.month - 1) // 3 + 1,
        price / (comp_price + 1e-9),          # price_ratio
        price * (1 - discount / 100),          # effective_price
        CATEGORY_MAP[product["category"]],
        REGION_MAP[product["region"]],
        WEATHER_MAP[product["weather"]],
        SEASON_MAP[product["season"]],
    ]])

    features_scaled = scaler.transform(features)
    state  = features_scaled[0]
    action = agent.act(state, training=False)

    base_cost  = price * 0.60
    new_price  = base_cost * MULTIPLIERS[action]
    eff_price  = new_price * (1 - discount / 100)

    return {
        "recommended_price": round(new_price, 2),
        "effective_price": round(eff_price, 2),
        "action": action,
        "multiplier": MULTIPLIERS[action],
    }


if __name__ == "__main__":
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
    print(result)