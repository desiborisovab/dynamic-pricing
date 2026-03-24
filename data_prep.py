import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

DATA_PATH = "data/retail_store_inventory.csv"

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    # Remove bad prices (free or extreme outliers)
    for col in ["price", "competitor_pricing"]:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] > 0) & (df[col].between(mean - 3 * std, mean + 3 * std))]

    # Drop rows with nulls
    df = df.dropna()
    return df.reset_index(drop=True)


def engineer_features(df):
    df = df.copy()

    # Date features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Price ratio vs competitor
    df["price_ratio"] = df["price"] / df["competitor_pricing"].replace(0, np.nan)

    # Effective price after discount
    df["effective_price"] = df["price"] * (1 - df["discount"] / 100)

    # Profit proxy (no cost data, so use price - rough margin assumption)
    df["revenue"] = df["effective_price"] * df["units_sold"]

    # Encode categoricals
    le = LabelEncoder()
    for col in ["category", "region", "weather_condition", "seasonality", "store_id"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    return df


def cluster_products(df):
    # Assign price tier A/B/C/D per category
    df = df.copy()
    df["price_tier"] = df.groupby("category")["price"].transform(
        lambda x: pd.qcut(x, 4, labels=["A", "B", "C", "D"], duplicates="drop")
    )
    df["cluster"] = df["category"] + "_" + df["price_tier"].astype(str)
    return df


def build_env_simulator(df):
    # Train a Random Forest to simulate demand (units_sold)
    # given state features + price action. Returns model + scaler + feature list.

    feature_cols = [
        "price", "competitor_pricing", "inventory_level",
        "discount", "holiday_promotion",
        "day_of_week", "month", "quarter",
        "price_ratio", "effective_price",
        "category_enc", "region_enc", "weather_condition_enc", "seasonality_enc"
    ]

    X = df[feature_cols].values
    y = df["units_sold"].values

    scaler = StandardScaler()

    # If used Linear Regression
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)

    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X, y)

    with open("sim_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    preds = model.predict(X_scaled)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"  Simulator  MAE={mae:.2f}  R²={r2:.3f}")

    return model, scaler, feature_cols


def prepare_all():
    print("[1/4] Loading & cleaning data...")
    df = load_and_clean()
    print(f" {len(df):,} rows retained")

    print("[2/4] Engineering features...")
    df = engineer_features(df)

    print("[3/4] Clustering products...")
    df = cluster_products(df)

    print("[4/4] Building environment simulator...")
    sim_model, sim_scaler, feature_cols = build_env_simulator(df)

    return df, sim_model, sim_scaler, feature_cols


if __name__ == "__main__":
    df, model, scaler, features = prepare_all()
    print(f"\nDataset shape : {df.shape}")
    print(f"Clusters : {df['cluster'].nunique()}")
    print(f"Products : {df['product_id'].nunique()}")
    print(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
