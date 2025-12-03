import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import numpy as np
import tkinter as tk
from tkinter import messagebox
import cv2
import torchvision.transforms as transforms


# ==== Model Definition ====
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


# ==== Labels ====
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ы", 32: "ь",
   33: "э", 34: "ю", 35: "я"
}

# ==== Load Model ====
model = ImprovedCNN(35)
model.load_state_dict(torch.load("model_cnn.pth", map_location="cpu"))
model.eval()


# ==== Preprocessing - MATCH TRAINING DATA ====
def preprocess_single_char(roi_binary):
    """
    Takes binary image (white char on black) and converts to 28x28 tensor
    matching how training data was prepared
    """
    # Convert to PIL
    img = Image.fromarray(roi_binary).convert("L")
    
    # Find bounding box to crop whitespace
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Resize to 20x20 (to match training prep - resize FIRST)
    img = img.resize((20, 20), Image.LANCZOS)
    
    # Pad to 28x28 (add 4 pixels on each side - black background)
    padded = Image.new("L", (28, 28), color=0)
    padded.paste(img, (4, 4))
    
    # Convert to tensor [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(padded).unsqueeze(0)
    return tensor, np.array(padded)


# ==== Drawing Application ====
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mongolian Character Recognition")
        
        info = tk.Label(root, text="Draw ONE character at a time | Draw THICK and BOLD", 
                       font=("Arial", 11), fg="yellow")
        info.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=400, height=400, bg='black', cursor="cross")
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (400, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Recognize", command=self.recognize,
                  bg='green', fg='white', width=15, font=("Arial", 12)).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text="Clear", command=self.clear,
                  bg='red', fg='white', width=15, font=("Arial", 12)).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="cyan")
        self.result_label.pack(pady=5)
        
        self.confidence_label = tk.Label(root, text="", font=("Arial", 12), fg="orange")
        self.confidence_label.pack(pady=5)

    def paint(self, event):
        # THICK brush - matches training style
        radius = 12
        x1, y1 = event.x - radius, event.y - radius
        x2, y2 = event.x + radius, event.y + radius
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (400, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    def recognize(self):
        img_array = np.array(self.image)
        
        # Binarize
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect any broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find the character region
        coords = cv2.findNonZero(binary)
        if coords is None:
            messagebox.showinfo("Error", "Draw something first!")
            return
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extract ROI
        roi = binary[y:y+h, x:x+w]
        
        # Preprocess (resize to 20x20, pad to 28x28)
        try:
            tensor, preview = preprocess_single_char(roi)
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {e}")
            return
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = nn.Softmax(dim=1)(outputs)
            conf, pred_idx = torch.max(probs, 1)
            char = LABELS.get(pred_idx.item() + 1, "?")
            confidence = conf.item() * 100
        
        self.result_label.config(text=f"Result: {char}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Show preprocessed 28x28 image
        preview_img = cv2.resize(preview, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Preprocessed 28x28 Image (what model sees)", preview_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==== Run ====
root = tk.Tk()
app = DrawApp(root)
root.mainloop()