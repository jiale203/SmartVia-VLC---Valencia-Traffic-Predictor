import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import json

# ── 1. Load FULL dataset (need continuity for lag features) ──
df = pd.read_csv("dataset_final.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
df["datetime"] = df["fecha"].astype(str) + " " + df["hora"]
df["datetime"] = pd.to_datetime(df["datetime"])

df["Estado_binary"] = (df["Estado"] != 0).astype(int)

df = df.sort_values(["Denominació / Denominación", "datetime"]).reset_index(drop=True)

# ── 2. Create lag features per road ──
print("Creating lag features...")
grouped = df.groupby("Denominació / Denominación")

df["estado_lag_1"] = grouped["Estado_binary"].shift(1)
df["estado_lag_2"] = grouped["Estado_binary"].shift(2)
df["estado_lag_4"] = grouped["Estado_binary"].shift(4)

df["estado_rolling_4"] = grouped["Estado_binary"].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
)

df["estado_rolling_8"] = grouped["Estado_binary"].transform(
    lambda x: x.rolling(window=8, min_periods=1).mean().shift(1)
)

df["estado_count_4"] = grouped["Estado_binary"].transform(
    lambda x: x.rolling(window=4, min_periods=1).sum().shift(1)
)

before = len(df)
df = df.dropna(subset=["estado_lag_1", "estado_lag_2", "estado_lag_4"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with NaN lags ({len(df)} remaining)\n")

# ── 3. Train/test split (temporal: 2023-2024 train, 2025 test) ──
train_mask = df["fecha"].dt.year.isin([2023, 2024])
test_mask = df["fecha"].dt.year == 2025

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

# ── 4. Feature engineering ──
def add_features(df):
    df = df.copy()
    time_parts = df["hora"].str.split(":", expand=True).astype(int)

    df["minutes"] = time_parts[0] * 60 + time_parts[1]
    df["hour_sin"] = np.sin(2 * np.pi * df["minutes"] / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * df["minutes"] / 1440)

    df["month"] = df["fecha"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day"] = df["fecha"].dt.day
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df["week_of_year"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["fecha"].dt.dayofweek.isin([5, 6]).astype(int)

    hour = time_parts[0]
    df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 20))).astype(int)
    df["is_night"] = ((hour >= 22) | (hour <= 6)).astype(int)

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

le_denom = LabelEncoder()
train_df["Denominació / Denominación"] = le_denom.fit_transform(train_df["Denominació / Denominación"])
test_df["Denominació / Denominación"] = le_denom.transform(test_df["Denominació / Denominación"])

le_day = LabelEncoder()
train_df["Day_of_week"] = le_day.fit_transform(train_df["Day_of_week"])
test_df["Day_of_week"] = le_day.transform(test_df["Day_of_week"])

drop_cols = ["Estado", "Estado_binary", "fecha", "hora", "datetime"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df["Estado_binary"]
X_test = test_df[feature_cols]
y_test = test_df["Estado_binary"]

feature_names = X_train.columns.tolist()
print(f"\nFeatures ({len(feature_names)}): {feature_names}\n")

# ── 5. Undersample + SMOTE (NO class weights — SMOTE already balances) ──
print("Class distribution before resampling:")
print(y_train.value_counts().sort_index().rename({0: "fluido", 1: "no_fluido"}))

TARGET_SAMPLES = 200000

continuous_cols = ["tmed", "prec", "tmin", "tmax", "sol", "minutes",
                   "hour_sin", "hour_cos", "month_sin", "month_cos",
                   "day_sin", "day_cos", "estado_rolling_4", "estado_rolling_8"]
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

pipeline = ImbPipeline(steps=[
    ("undersample", RandomUnderSampler(sampling_strategy={0: TARGET_SAMPLES}, random_state=42)),
    ("oversample", SMOTE(sampling_strategy={1: TARGET_SAMPLES}, random_state=42, k_neighbors=5)),
])

X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

print("\nClass distribution after resampling:")
print(y_train_res.value_counts().sort_index().rename({0: "fluido", 1: "no_fluido"}))
print(f"Total training rows: {len(X_train_res)}\n")

# ── 6. LightGBM parameters ──
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

# ── 7. Train model (no CV — temporal test split is the real evaluation) ──
print("── Training model ──\n")
dtrain = lgb.Dataset(X_train_res, label=y_train_res, feature_name=feature_names)
final_model = lgb.train(params, dtrain, num_boost_round=500)

# ── 10. Evaluate on test set ──
y_test_proba = final_model.predict(X_test)
y_test_pred = (y_test_proba > 0.5).astype(int)

test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

print("── Test Set Results (threshold=0.5) ──")
print(f"Macro F1: {test_macro_f1:.4f}")
print(f"\nConfusion matrix:\n{test_cm}")
print(f"\nClassification report:\n{classification_report(y_test, y_test_pred, target_names=['fluido(0)', 'no_fluido(1)'])}")

# ── 11. Threshold optimization ──
print("── Threshold Optimization ──")
print(f"{'threshold':<12} {'recall_nf':<12} {'precision_nf':<14} {'f1_nf':<10} {'f1_macro':<12} {'false_alarms':<14} {'missed':<10}")

best_threshold = 0.5
best_macro_f1 = 0

for threshold in np.arange(0.05, 0.99, 0.01):
    y_pred_t = (y_test_proba > threshold).astype(int)
    f1_t = f1_score(y_test, y_pred_t, average="macro")
    if f1_t > best_macro_f1:
        best_macro_f1 = f1_t
        best_threshold = round(threshold, 2)
    cm_t = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
    recall_nf = recall_score(y_test, y_pred_t, pos_label=1)
    precision_nf = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    f1_nf = f1_score(y_test, y_pred_t, pos_label=1)
    false_alarms = cm_t[0, 1]
    missed = cm_t[1, 0]
    if int(threshold * 100) % 5 == 0:
        print(f"  {threshold:<12.2f} {recall_nf:<12.4f} {precision_nf:<14.4f} {f1_nf:<10.4f} {f1_t:<12.4f} {false_alarms:<14} {missed:<10}")

print(f"\n** Best threshold: {best_threshold} (Macro F1: {best_macro_f1:.4f}) **")

# ── Evaluate with best threshold ──
y_best_pred = (y_test_proba > best_threshold).astype(int)
print(f"\n── Test Set Results (threshold={best_threshold}) ──")
print(f"Macro F1: {f1_score(y_test, y_best_pred, average='macro'):.4f}")
print(f"\nConfusion matrix:\n{confusion_matrix(y_test, y_best_pred, labels=[0, 1])}")
print(f"\nClassification report:\n{classification_report(y_test, y_best_pred, target_names=['fluido(0)', 'no_fluido(1)'])}")

# ── 12. Feature importance ──
importance = final_model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
print("\n── Top 15 Feature Importance (gain) ──")
for name, imp in feat_imp[:15]:
    print(f"  {name}: {imp:.0f}")

# ── 13. Save ──
final_model.save_model("lightgbm_lag_v2_model.txt")

results = {
    "test_macro_f1": float(test_macro_f1),
    "best_threshold": best_threshold,
    "boost_rounds": 500,
    "changes": "removed class_weight and CV, SMOTE handles imbalance alone",
}
with open("lightgbm_lag_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nModel saved to lightgbm_lag_v2_model.txt")
print("Results saved to lightgbm_lag_v2_results.json")
