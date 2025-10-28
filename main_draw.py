import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# ==== Load Model ====
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

model = ImprovedCNN(35)
model.load_state_dict(torch.load("model_cnn.pth", map_location="cpu"))
model.eval()

LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ы", 32: "ь",
   33: "э", 34: "ю", 35: "я"
}

# ==== Drawing Window ====
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Cyrillic Character - FIXED!")
        
        # BLACK canvas (like training data!)
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack()
        
        # Black background image
        self.image = Image.new("L", (280, 280), 0)  # 0 = black
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Recognize", command=self.recognize_debug, bg='green', fg='white', width=15).pack(side=tk.LEFT, padx=3, pady=5)
        tk.Button(btn_frame, text="Clear", command=self.clear, bg='red', fg='white', width=15).pack(side=tk.LEFT, padx=3, pady=5)
        
        self.result_label = tk.Label(root, text="Draw WHITE on BLACK (like training data)", font=("Arial", 12))
        self.result_label.pack(pady=5)
        
    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        # Draw WHITE on black (like training data!)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)  # 255 = white
    
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)  # 0 = black
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw WHITE on BLACK (like training data)")
    
    def preprocess(self):
        img = self.image.copy()
        bbox = img.getbbox()
        if bbox is None:
            return None
        
        img = img.crop(bbox)
        w, h = img.size
        new_size = max(w, h) + 40
        new_img = Image.new("L", (new_size, new_size), 0)  # 0 = black background
        new_img.paste(img, ((new_size - w) // 2, (new_size - h) // 2))
        
        # NO invert - already correct colors!
        img = ImageEnhance.Contrast(new_img).enhance(2.0)
        threshold = 128
        img = img.point(lambda p: 255 if p > threshold else 0)
        img = img.resize((28, 28), Image.LANCZOS)
        
        return img
    
    def recognize_debug(self):
        processed_img = self.preprocess()
        if processed_img is None:
            messagebox.showinfo("Error", "Draw something first!")
            return
        
        processed_img.save("debug_your_drawing.png")
        
        img_array = np.array(processed_img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = nn.Softmax(dim=1)(outputs)
            
            # Get TOP 5 predictions
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            predictions_text = "TOP 5:\n"
            for i in range(5):
                char = LABELS.get(top5_indices[0][i].item() + 1, "?")
                conf = top5_probs[0][i].item() * 100
                predictions_text += f"{i+1}. {char}: {conf:.1f}%\n"
            
            # Best prediction
            best_char = LABELS.get(top5_indices[0][0].item() + 1, "?")
            best_conf = top5_probs[0][0].item() * 100
        
        # Show comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(processed_img, cmap='gray')
        axes[0].set_title(f"Your Drawing\n(WHITE on BLACK)", fontsize=14)
        axes[0].axis('off')
        
        axes[1].text(0.5, 0.5, predictions_text, 
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgreen'))
        axes[1].set_title(f"Prediction: {best_char} ({best_conf:.1f}%)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        self.result_label.config(text=f"Predicted: {best_char} ({best_conf:.1f}%)")

# ==== Run ====
root = tk.Tk()
app = DrawApp(root)
root.mainloop()