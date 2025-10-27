import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from cnn_model import ImprovedCNN as CNNModel  # your CNN model file

# ======== SETTINGS ========
csv_path = "C:/Projects/python/AI/character dataset/HMCC letters merged.csv"
model_save_path = "C:/Projects/python/AI/model_cnn.pth"
batch_size = 64
epochs = 30
patience = 5
learning_rate = 0.001

# ======== CUSTOM DATASET ========
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file).values
        self.X = torch.tensor(data[:, 1:], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        self.y = torch.tensor(data[:, 0], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CSVDataset(csv_path)

# ======== SPLIT TRAIN / VALIDATION ========
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ======== MODEL SETUP ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=len(torch.unique(dataset.y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======== METRICS ========
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = np.inf
no_improve_count = 0

# ======== LIVE PLOTTING ========
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ======== TRAIN LOOP ========
for epoch in range(1, epochs + 1):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ===== Validation =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy)

    # ===== Print progress =====
    print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}% | "
          f"Time: {(time.time()-start_time):.1f}s")

    # ===== Live Graph =====
    ax1.clear()
    ax2.clear()
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='orange')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.pause(0.1)

    # ===== Early Stopping =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

plt.ioff()
plt.show()
print(f"💾 Model improved and saved at epoch {epoch} (Val Loss: {avg_val_loss:.4f})")

