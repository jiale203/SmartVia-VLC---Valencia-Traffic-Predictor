import pandas as pd

df = pd.read_csv("dataset_final.csv")
df["fecha"] = pd.to_datetime(df["fecha"])

train_mask = df["fecha"].dt.year.isin([2023, 2024])
test_mask = df["fecha"].dt.year == 2025

train_df = df[train_mask]
test_df = df[test_mask]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Train: {len(train_df)} rows, {train_df['fecha'].dt.date.nunique()} unique days")
print(f"Test:  {len(test_df)} rows, {test_df['fecha'].dt.date.nunique()} unique days")
