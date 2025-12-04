import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import messagebox
import cv2
import torchvision.transforms as transforms


# ==========================
#       MODEL DEFINITION
# ==========================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
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


# ==========================
#            LABELS
# ==========================
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ы", 32: "ь",
   33: "э", 34: "ю", 35: "я"
}


# ==========================
#      LOAD MODEL
# ==========================
model = ImprovedCNN(35)
model.load_state_dict(torch.load("model_cnn.pth", map_location="cpu"))
model.eval()


# ==========================
#    PREPROCESS CHARACTER
# ==========================
def preprocess_single_char(roi_binary):
    img = Image.fromarray(roi_binary).convert("L")

    # bounding box crop
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img = img.resize((20, 20), Image.LANCZOS)

    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, (4, 4))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(padded).unsqueeze(0)


# ==========================
#         DRAW APP
# ==========================
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mongolian Multi-Character Recognition (Projection Profile)")

        info = tk.Label(root, text="Draw characters (WHITE on BLACK). Separate them by SPACE.",
                        font=("Arial", 11), fg="yellow")
        info.pack(pady=5)

        self.canvas = tk.Canvas(root, width=1200, height=400, bg='black', cursor="cross")
        self.canvas.pack(pady=10)

        # PIL image
        self.image = Image.new("L", (1200, 400), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Track previous mouse position
        self.last_x = None
        self.last_y = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Recognize All", command=self.recognize,
                  bg='green', fg='white', width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear,
                  bg='red', fg='white', width=20).pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 20, "bold"))
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(root, text="", font=("Arial", 11))
        self.confidence_label.pack(pady=5)

    # ===== SMOOTH PAINT =====
    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Tkinter draw
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=22, capstyle="round", smooth=True
            )

            # PIL draw
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=22
            )

        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (1200, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    # ===== RECOGNITION =====
    def recognize(self):
        img_array = np.array(self.image)

        # threshold
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)

        # === Horizontal Projection Profile
        col_sum = np.sum(binary == 255, axis=0)

        segments = []
        start = None

        for x in range(len(col_sum)):
            if col_sum[x] > 0 and start is None:
                start = x
            if col_sum[x] == 0 and start is not None:
                end = x
                if end - start > 5:
                    segments.append((start, end))
                start = None

        if start is not None:
            segments.append((start, len(col_sum)))

        if not segments:
            messagebox.showinfo("Error", "No characters found!")
            return

        recognized_text = ""
        confidence_list = []

        for (x1, x2) in segments:
            roi = binary[:, x1:x2]

            row_sum = np.sum(roi == 255, axis=1)
            top = np.argmax(row_sum > 0)
            bottom = len(row_sum) - np.argmax(row_sum[::-1] > 0)
            roi = roi[top:bottom, :]

            tensor = preprocess_single_char(roi)

            with torch.no_grad():
                output = model(tensor)
                probs = nn.Softmax(dim=1)(output)
                conf, pred = torch.max(probs, 1)

            char = LABELS[pred.item() + 1]
            recognized_text += char
            confidence_list.append(f"{char}({conf.item()*100:.0f}%)")

        self.result_label.config(text=f"Result: {recognized_text}")
        self.confidence_label.config(text=" | ".join(confidence_list))


# ==========================
#          RUN APP
# ==========================
root = tk.Tk()
app = DrawApp(root)
root.mainloop()
