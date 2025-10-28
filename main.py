import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cnn_model import ImprovedCNN as CNNModel
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==== Paths ====
model_path = "C:/Projects/python/AI/model_cnn.pth"
image_folder = "C:/Projects/python/ai/images"

# ==== Labels Mapping ====
labels_map = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ы", 32: "ь",
   33: "э", 34: "ю", 35: "я"
}

# ==== Load Model ====
num_classes = 35
model = CNNModel(num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ==== Improved Preprocessing ====
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")              # Grayscale
    img = ImageOps.invert(img)                           # Invert colors
    img = ImageEnhance.Contrast(img).enhance(2.0)        # Boost contrast

    # Optional: Binarize to pure black & white
    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)

    img = img.resize((28, 28))                           # Resize

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, np.array(img)

# ==== Load Images ====
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

true_labels, pred_labels, confidences, images_np = [], [], [], []

for img_file in image_files:
    true_char = os.path.splitext(img_file)[0][0]
    true_labels.append(true_char)

    img_path = os.path.join(image_folder, img_file)
    img_tensor, img_np = preprocess_image(img_path)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        conf, pred_idx = torch.max(probs, 1)
        pred_char = labels_map.get(pred_idx.item() + 1, "?")
        pred_labels.append(pred_char)
        confidences.append(conf.item() * 100)

    images_np.append(img_np)

# ==== Accuracy Summary ====
correct = sum(t == p for t, p in zip(true_labels, pred_labels))
total = len(true_labels)
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n✅ Correct Predictions: {correct}/{total} ({accuracy:.2f}%)")

# ==== Show All Images in Grid with T/P/Confidence ====
num_images = len(images_np)
cols = 5
rows = (num_images + cols - 1) // cols
plt.figure(figsize=(cols * 3, rows * 3))

for i, (img_np, t, p, c) in enumerate(zip(images_np, true_labels, pred_labels, confidences)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    color = "green" if t == p else "red"
    plt.title(f"T:{t}\nP:{p} {c:.1f}%", fontsize=10, color=color)

plt.tight_layout()
plt.show()

# ==== Confusion Matrix ====
cm = confusion_matrix(true_labels, pred_labels, labels=list(labels_map.values()))
plt.figure(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels_map.values()))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, colorbar=False)
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f}%)")
plt.show()





