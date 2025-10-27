import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# ==== 1. Load CSV Files ====
train_path = "C:/Projects/python/ai/dataset/train_data.csv"
test_path = "C:/Projects/python/ai/dataset/test_data.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Keep first column for label mapping
train_labels_numeric = train_df.iloc[:, 0].values.astype(np.int64)

# ==== 2. Prepare Training Data ====
# CNN input: only pixel columns (exclude first column)
X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
y_train = torch.tensor(train_labels_numeric)

X_train = torch.tensor(X_train).reshape(-1, 1, 28, 28)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ==== 3. Prepare Test Data ====
# Keep first column in test for reference if exists
# CNN input: pixel columns
if test_df.shape[1] > 784:  # assuming first column is label in test
    test_labels_numeric = test_df.iloc[:, 0].values
    X_test_pixels = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
else:
    X_test_pixels = test_df.values.astype(np.float32) / 255.0

X_test = torch.tensor(X_test_pixels).reshape(-1, 1, 28, 28)
test_dataset = TensorDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# ==== 4. Define Improved CNN Model ====
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        # Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==== 5. Initialize Model, Loss, and Optimizer ====
num_classes = len(np.unique(y_train))
model = ImprovedCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

# ==== 6. Train the Model ====
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ==== 7. Generate Predictions on Test Data ====
model.eval()
predictions_numeric = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch[0]
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions_numeric.extend(predicted.tolist())

# ==== 8. Map numeric predictions to characters (optional) ====
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ь", 32: "ы",
   33: "э", 34: "ю", 35: "я"
}

predictions_char = [LABELS.get(x+1, "?") for x in predictions_numeric]

# ==== 9. Save Submission ====
submission_df = pd.DataFrame({
    "numeric_label": predictions_numeric,
    "character_label": predictions_char
})
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")

# ==== 10. Save Model ====
torch.save(model.state_dict(), "model_cnn.pth")
print("✅ Model saved as model_cnn.pth")