import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load training data
train_df = pd.read_csv("C:/Projects/python/ai/dataset/train_data.csv")

# Find samples of а, у, х, ш
chars_to_check = {1: "а", 22: "у", 25: "х", 28: "ш"}

fig, axes = plt.subplots(4, 5, figsize=(12, 10))

for row, (label_num, char) in enumerate(chars_to_check.items()):
    samples = train_df[train_df.iloc[:, 0] == label_num].iloc[:5]
    
    for col in range(min(5, len(samples))):
        pixels = samples.iloc[col, 1:].values.reshape(28, 28)
        axes[row, col].imshow(pixels, cmap='gray')
        axes[row, col].set_title(f"{char} (label {label_num})")
        axes[row, col].axis('off')

plt.suptitle("Training Data Samples - Check if they look correct!", fontsize=16)
plt.tight_layout()
plt.show()