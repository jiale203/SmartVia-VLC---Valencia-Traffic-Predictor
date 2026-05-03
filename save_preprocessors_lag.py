import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("Loading dataset_final.csv...")
df = pd.read_csv("dataset_final.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
df["Estado_binary"] = (df["Estado"] != 0).astype(int)

time_parts = df["hora"].str.split(":", expand=True).astype(int)
df["minute_slot"] = time_parts[0] * 60 + time_parts[1]
df["day_of_week"] = df["fecha"].dt.dayofweek

train_df = df[df["fecha"].dt.year.isin([2023, 2024])].copy()

# ── 1. Historical lookup tables ──
print("Building historical lookup tables...")
historical = train_df.groupby(
    ["Denominació / Denominación", "day_of_week", "minute_slot"]
)["Estado_binary"].mean().reset_index()
historical.columns = ["road", "day_of_week", "minute_slot", "hist_prob"]

historical_fallback = train_df.groupby(
    ["Denominació / Denominación", "minute_slot"]
)["Estado_binary"].mean().reset_index()
historical_fallback.columns = ["road", "minute_slot", "hist_prob_fallback"]

joblib.dump(historical, "historical.joblib")
joblib.dump(historical_fallback, "historical_fallback.joblib")

# ── 2. Label encoders ──
le_denom = LabelEncoder()
le_denom.fit(train_df["Denominació / Denominación"].unique())

le_day = LabelEncoder()
le_day.fit(train_df["Day_of_week"].unique())

joblib.dump(le_denom, "le_denom.joblib")
joblib.dump(le_day, "le_day.joblib")
joblib.dump(sorted(le_denom.classes_.tolist()), "known_roads.joblib")

# ── 3. Scaler (with lag features) ──
print("Computing lag features for scaler...")
train_full = train_df.sort_values(["Denominació / Denominación", "fecha", "hora"]).reset_index(drop=True)
grouped = train_full.groupby("Denominació / Denominación")
train_full["estado_rolling_4"] = grouped["Estado_binary"].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
)
train_full["estado_rolling_8"] = grouped["Estado_binary"].transform(
    lambda x: x.rolling(window=8, min_periods=1).mean().shift(1)
)
train_full = train_full.dropna(subset=["estado_rolling_4", "estado_rolling_8"])

time_parts_train = train_full["hora"].str.split(":", expand=True).astype(int)
minutes_train = time_parts_train[0] * 60 + time_parts_train[1]

scaler_df = pd.DataFrame({
    "tmed": train_full["tmed"].values,
    "prec": train_full["prec"].values,
    "tmin": train_full["tmin"].values,
    "tmax": train_full["tmax"].values,
    "sol": train_full["sol"].values,
    "minutes": minutes_train.values,
    "hour_sin": np.sin(2 * np.pi * minutes_train.values / 1440),
    "hour_cos": np.cos(2 * np.pi * minutes_train.values / 1440),
    "month_sin": np.sin(2 * np.pi * train_full["fecha"].dt.month.values / 12),
    "month_cos": np.cos(2 * np.pi * train_full["fecha"].dt.month.values / 12),
    "day_sin": np.sin(2 * np.pi * train_full["fecha"].dt.day.values / 31),
    "day_cos": np.cos(2 * np.pi * train_full["fecha"].dt.day.values / 31),
    "estado_rolling_4": train_full["estado_rolling_4"].values,
    "estado_rolling_8": train_full["estado_rolling_8"].values,
})
scaler_lag = StandardScaler()
scaler_lag.fit(scaler_df)

joblib.dump(scaler_lag, "scaler_lag.joblib")

print(f"\nSaved: historical.joblib, historical_fallback.joblib")
print(f"Saved: le_denom.joblib, le_day.joblib, known_roads.joblib")
print(f"Saved: scaler_lag.joblib")
print(f"Known roads: {len(le_denom.classes_)}")
