import pandas as pd
import numpy as np

# === Paths ===
train_path = "C:/Projects/python/ai/dataset/train_data.csv"
test_path = "C:/Projects/python/ai/dataset/test_data.csv"

# === Load CSVs ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === Basic info ===
print("=== DATASET INFO ===")
print(f"Train file: {train_path}")
print(f"  Rows (images): {train_df.shape[0]}")
print(f"  Columns (pixels per image): {train_df.shape[1]}")

print(f"\nTest file: {test_path}")
print(f"  Rows (images): {test_df.shape[0]}")
print(f"  Columns (pixels per image): {test_df.shape[1]}")

# === Check for negatives ===
train_neg_count = (train_df.values < 0).sum()
test_neg_count = (test_df.values < 0).sum()

train_min = train_df.values.min()
test_min = test_df.values.min()

print("\n=== NEGATIVE VALUE CHECK ===")
print(f"Train file:")
print(f"  Contains negatives? {'❌ Yes' if train_neg_count > 0 else '✅ No'}")
print(f"  Total negative pixels: {train_neg_count}")
print(f"  Minimum pixel value: {train_min}")

print(f"\nTest file:")
print(f"  Contains negatives? {'❌ Yes' if test_neg_count > 0 else '✅ No'}")
print(f"  Total negative pixels: {test_neg_count}")
print(f"  Minimum pixel value: {test_min}")

# === Optional: fix negatives ===
if train_neg_count > 0 or test_neg_count > 0:
    print("\n🛠️ Fixing negative values (replacing with 0)...")
    train_df[train_df < 0] = 0
    test_df[test_df < 0] = 0

    # Save cleaned files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("✅ Negative values fixed and saved.")
else:
    print("\n✅ No negative values found — dataset is clean.")
