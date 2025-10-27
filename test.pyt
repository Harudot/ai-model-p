import pandas as pd

# Load your CSV
data = pd.read_csv("C:/Projects/python/ai/character dataset/HMCC letters merged.csv")

# Separate features and labels
# Assuming first column is label
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# For checking, let's call X_test the same as X (or you can load your real test CSV)
X_test = X

# Now check shape
num_samples = X_test.shape[0]  # number of rows
num_features = X_test.shape[1]  # number of columns
print("Test samples:", num_samples)
print("Test features:", num_features)

expected_features = 28*28
print("Expected features per image:", expected_features)

if num_features != expected_features:
    print("⚠️ Warning: Number of features does not match expected image size!")
