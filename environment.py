"""
Dynamic Pricing Environment
Wraps the simulator so the DQN agent can interact with it step-by-step.
"""

import numpy as np


# 10 actions: multiply cost by 1.00, 1.025, 1.05, ... 1.225  (2.5% increments)
N_ACTIONS   = 10
MULTIPLIERS = np.array([1.00 + 0.025 * i for i in range(N_ACTIONS)])


class PricingEnv:
    """
    State  : feature vector for the current product/time snapshot
    Action : index 0-9  →  price = base_cost * MULTIPLIERS[action]
    Reward : profit = (effective_price - base_cost) * predicted_demand
    """

    def __init__(self, df, sim_model, sim_scaler, feature_cols, seed=42):
        self.df          = df.reset_index(drop=True)
        self.sim_model   = sim_model
        self.sim_scaler  = sim_scaler
        self.feature_cols = feature_cols
        self.rng         = np.random.default_rng(seed)
        self.n_features  = len(feature_cols)
        self.n_actions   = N_ACTIONS
        self._idx        = 0
        self._order      = self.rng.permutation(len(self.df))

    # ------------------------------------------------------------------
    def reset(self):
        self._order = self.rng.permutation(len(self.df))
        self._idx   = 0
        return self._get_state()

    def step(self, action: int):
        row          = self.df.iloc[self._order[self._idx]]
        base_cost    = row["price"] * 0.6          # assume 60 % cost ratio
        multiplier   = MULTIPLIERS[action]
        new_price    = base_cost * multiplier
        discount_pct = row["discount"]
        eff_price    = new_price * (1 - discount_pct / 100)

        # Build feature row with new price
        feat = row[self.feature_cols].copy()
        feat["price"]           = new_price
        feat["effective_price"] = eff_price
        feat["price_ratio"]     = new_price / (row["competitor_pricing"] + 1e-9)

        x_scaled = self.sim_scaler.transform(feat.values.reshape(1, -1))
        pred_demand = max(0, float(self.sim_model.predict(x_scaled)[0]))

        profit  = (eff_price - base_cost) * pred_demand
        reward  = profit / 1000.0          # scale reward

        self._idx += 1
        done       = self._idx >= len(self.df)
        next_state = self._get_state() if not done else np.zeros(self.n_features)

        info = dict(
            price=new_price, demand=pred_demand,
            profit=profit, multiplier=multiplier,
            product_id=row["product_id"], date=str(row["date"].date())
        )
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def _get_state(self):
        if self._idx >= len(self.df):
            return np.zeros(self.n_features)
        row  = self.df.iloc[self._order[self._idx]]
        feat = row[self.feature_cols].values.astype(float)
        return feat

    def baseline_profit(self):
        """Compute total profit using original prices (no RL)"""
        total = 0.0
        for _, row in self.df.iterrows():
            base_cost  = row["price"] * 0.6
            eff_price  = row["price"] * (1 - row["discount"] / 100)
            feat       = row[self.feature_cols].values.reshape(1, -1)
            x_scaled   = self.sim_scaler.transform(feat)
            demand     = max(0, float(self.sim_model.predict(x_scaled)[0]))
            total     += (eff_price - base_cost) * demand
        return total