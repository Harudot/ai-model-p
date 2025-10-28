import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import tkinter as tk

# ==== Model Definition (same as training) ====
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 256),
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

# ==== Load Model ====
num_classes = 35
model = ImprovedCNN(num_classes)
model.load_state_dict(torch.load("C:/Projects/python/AI/model_cnn.pth", map_location="cpu"))
model.eval()

# ==== Labels ====
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ь", 32: "ы",
   33: "э", 34: "ю", 35: "я"
}

# ==== Preprocess Drawing ====
def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert colors (white on black)
    img = cv2.bitwise_not(img)
    # Threshold / Binarize
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # Find bounding box of content
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    # Resize to 20x20 and pad to 28x28
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(img, ((4,4),(4,4)), mode='constant', constant_values=0)
    # Convert to tensor
    tensor = transforms.ToTensor()(padded).unsqueeze(0)
    return tensor

# ==== Predict Function ====
def predict(img):
    with torch.no_grad():
        tensor = preprocess_image(img)
        outputs = model(tensor)
        probs = nn.Softmax(dim=1)(outputs)
        conf, idx = torch.max(probs, 1)
        return LABELS[idx.item()+1], conf.item()*100

# ==== Tkinter Drawing ====
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a character")
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()
        self.image = np.zeros((280,280,3), dtype=np.uint8)
        self.canvas.bind("<B1-Motion>", self.paint)
        btn = tk.Button(self.root, text="Predict", command=self.on_predict)
        btn.pack()
        self.label = tk.Label(self.root, text="Draw and click Predict", font=("Arial",16))
        self.label.pack()
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = event.x-10, event.y-10
        x2, y2 = event.x+10, event.y+10
        cv2.circle(self.image, (event.x, event.y), 10, (255,255,255), -1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

    def on_predict(self):
        char, conf = predict(self.image)
        self.label.config(text=f"🧠 Predicted: {char} ({conf:.1f}%)")

# ==== Run App ====
App()


