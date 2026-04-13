# run this once in your terminal
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/retail_store_inventory.csv")
le = LabelEncoder()
le.fit(df["Seasonality"].astype(str))
print(dict(zip(le.classes_, le.transform(le.classes_))))