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


# ==== Preprocessing for single character ====
def preprocess_single_char(roi_binary):
    """
    Takes binary image (white char on black) and converts to 28x28 tensor
    """
    # Convert to PIL
    img = Image.fromarray(roi_binary).convert("L")
    
    # Find bounding box to crop whitespace
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Resize to 20x20
    img = img.resize((20, 20), Image.LANCZOS)
    
    # Pad to 28x28 (add 4 pixels on each side - black background)
    padded = Image.new("L", (28, 28), color=0)
    padded.paste(img, (4, 4))
    
    # Convert to tensor [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(padded).unsqueeze(0)
    return tensor


# ==== Drawing Application ====
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mongolian Multi-Character Recognition")
        
        info = tk.Label(root, text="Draw multiple characters (WHITE on BLACK) | Space them out clearly!", 
                       font=("Arial", 11), fg="yellow")
        info.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=1200, height=400, bg='black', cursor="cross")
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (1200, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Recognize All", command=self.recognize,
                  bg='green', fg='white', width=20, font=("Arial", 12)).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text="Clear", command=self.clear,
                  bg='red', fg='white', width=20, font=("Arial", 12)).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), fg="black")
        self.result_label.pack(pady=10)
        
        self.confidence_label = tk.Label(root, text="", font=("Arial", 11), fg="black")
        self.confidence_label.pack(pady=5)

    def paint(self, event):
        # THICK brush
        radius = 12
        x1, y1 = event.x - radius, event.y - radius
        x2, y2 = event.x + radius, event.y + radius
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (1200, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    def recognize(self):
        img_array = np.array(self.image)
        
        # Binarize
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # STRONG dilation to connect broken strokes and separate parts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        binary = cv2.dilate(binary, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find connected components (each character)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            messagebox.showinfo("Error", "Draw something first!")
            return
        
        # Extract valid character regions
        regions = []
        MIN_AREA = 200
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter noise
            if area < MIN_AREA or h < 10 or w < 5:
                continue
            
            regions.append({
                'x': x,
                'y': y,
                'x2': x + w,
                'y2': y + h,
                'w': w,
                'h': h,
                'center_x': centroids[i][0],
                'center_y': centroids[i][1]
            })
        
        # Merge regions that are close together (within 20 pixels)
        merged = []
        used = set()
        
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            
            # Start a new merged region
            merged_region = {
                'x': r1['x'],
                'y': r1['y'],
                'x2': r1['x2'],
                'y2': r1['y2']
            }
            used.add(i)
            
            # Find nearby regions to merge
            for j, r2 in enumerate(regions):
                if j <= i or j in used:
                    continue
                
                # Check if regions are close (horizontal or vertical gap < 20 pixels)
                h_gap = max(0, max(r1['x'], r2['x']) - min(r1['x2'], r2['x2']))
                v_gap = max(0, max(r1['y'], r2['y']) - min(r1['y2'], r2['y2']))
                
                if h_gap < 20 and v_gap < 20:
                    # Merge this region
                    merged_region['x'] = min(merged_region['x'], r2['x'])
                    merged_region['y'] = min(merged_region['y'], r2['y'])
                    merged_region['x2'] = max(merged_region['x2'], r2['x2'])
                    merged_region['y2'] = max(merged_region['y2'], r2['y2'])
                    used.add(j)
            
            merged.append(merged_region)
        
        # Convert merged regions to character data
        characters = []
        for region in merged:
            x = region['x']
            y = region['y']
            w = region['x2'] - region['x']
            h = region['y2'] - region['y']
            
            # Extract ROI from binary
            roi = binary[y:y+h, x:x+w]
            
            characters.append({
                'roi': roi,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': (region['x'] + region['x2']) / 2
            })
        
        if not characters:
            messagebox.showinfo("Error", "No characters detected!")
            return
        
        # Sort characters left to right
        characters.sort(key=lambda c: c['center_x'])
        
        # Predict each character
        recognized_text = ""
        confidences = []
        
        for char_data in characters:
            roi = char_data['roi']
            
            try:
                tensor = preprocess_single_char(roi)
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
                
                recognized_text += char
                confidences.append(f"{char}({confidence:.0f}%)")
        
        # Display results
        self.result_label.config(text=f"Result: {recognized_text}")
        self.confidence_label.config(text="  |  ".join(confidences))
        
        # Visualize detected regions with green boxes
        img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        for idx, char_data in enumerate(characters):
            x, y, w, h = char_data['x'], char_data['y'], char_data['w'], char_data['h']
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_color, str(idx+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Detected {len(characters)} Characters", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==== Run ====
root = tk.Tk()
app = DrawApp(root)
root.mainloop()