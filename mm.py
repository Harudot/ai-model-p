import torch
import pandas as pd
import numpy as np
from cnn_model import CNNModel  # import the class

# ==== 1. Load trained model ====
num_classes = 35  # total number of Mongolian characters
model = CNNModel(num_classes)
model.load_state_dict(torch.load("C:/Projects/python/AI/model_cnn.pth"))
model.eval()

# ==== 2. Load Test Data ====
test_path = "C:/Projects/python/ai/dataset/test_data.csv"
test_df = pd.read_csv(test_path)

# If test CSV has a first column as numeric labels
if test_df.shape[1] > 784:
    y_test = torch.tensor(test_df.iloc[:, 0].values.astype(np.int64))
    X_test_pixels = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
else:
    y_test = None
    X_test_pixels = test_df.values.astype(np.float32) / 255.0

X_test = torch.tensor(X_test_pixels).reshape(-1, 1, 28, 28)
test_dataset = torch.utils.data.TensorDataset(X_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# ==== 3. Make Predictions ====
predictions_numeric = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch[0]  # TensorDataset returns tuple
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions_numeric.extend(predicted.tolist())

# ==== 4. Compute Test Accuracy (if labels exist) ====
if y_test is not None:
    correct = sum([p == t for p, t in zip(predictions_numeric, y_test.tolist())])
    total = len(y_test)
    print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
else:
    print("⚠️ No test labels found, accuracy cannot be computed.")

# ==== 5. Map numeric predictions to Mongolian characters ====
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ь", 32: "ы",
   33: "э", 34: "ю", 35: "я"
}
predictions_char = [LABELS.get(x+1, "?") for x in predictions_numeric]

# ==== 6. Save Submission CSV ====
submission_df = pd.DataFrame({
    "numeric_label": predictions_numeric,
    "character_label": predictions_char
})
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")

