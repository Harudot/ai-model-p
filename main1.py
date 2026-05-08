import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
import os

# -------------------
# Labels
# -------------------
LABELS = {
   1: "а", 2: "б", 3: "в", 4: "г", 5: "д", 6: "е", 7: "ё", 8: "ж",
   9: "з", 10: "и", 11: "й", 12: "к", 13: "л", 14: "м", 15: "н", 16: "о",
   17: "ө", 18: "п", 19: "р", 20: "с", 21: "т", 22: "у", 23: "ү", 24: "ф",
   25: "х", 26: "ц", 27: "ч", 28: "ш", 29: "щ", 30: "ъ", 31: "ь", 32: "ы",
   33: "э", 34: "ю", 35: "я"
}
NUM_CLASSES = 35

# -------------------
# CNN model
# -------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# -------------------
# Connected Components using DFS (SIMPLIFIED - no background marking)
# -------------------
def find_connected_components(binary_img, threshold=128):
    """Find connected components manually using DFS
    
    Simple approach: just find all connected white pixels.
    Don't worry about internal holes - let merging handle it.
    """
    H, W = binary_img.shape
    visited = np.zeros_like(binary_img, dtype=bool)
    components = []
    
    def dfs(y, x, current):
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= H or cx < 0 or cx >= W:
                continue
            if visited[cy, cx] or binary_img[cy, cx] <= threshold:
                continue
            visited[cy, cx] = True
            current.append((cy, cx))
            # 8-connected neighbors
            for ny in range(cy-1, cy+2):
                for nx in range(cx-1, cx+2):
                    if (ny, nx) != (cy, cx):
                        stack.append((ny, nx))
    
    # Scan every pixel
    for y in range(H):
        for x in range(W):
            if binary_img[y, x] > threshold and not visited[y, x]:
                current = []
                dfs(y, x, current)
                if current:
                    components.append(current)
    
    return components

def components_to_boxes(components, min_size=30):
    """Convert components to bounding boxes, filtering by size"""
    boxes = []
    
    for comp in components:
        if len(comp) < min_size:  # Filter out noise
            continue
        
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        
        x1, x2 = min(xs), max(xs) + 1
        y1, y2 = min(ys), max(ys) + 1
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes

def merge_close_boxes(boxes):
    """Merge boxes using area-based filtering with smart vertical alignment
    
    Logic:
    1. Use FIXED size thresholds
    2. Boxes > 5000 px = main characters
    3. Boxes <= 5000 px = parts (diacritics like ё dots, й dot, etc)
    4. Merge parts ONLY if:
       - They are ABOVE or BELOW the main character (vertically stacked)
       - They are within the horizontal bounds of the character
       - This prevents merging with nearby separate characters
    """
    if not boxes:
        return []
    
    # FIXED thresholds
    MAIN_CHAR_THRESHOLD = 5000  # Pixels
    
    # Calculate areas
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    
    print(f"Box areas: {sorted(areas)}")
    print(f"Main character threshold: {MAIN_CHAR_THRESHOLD} px")
    print()
    
    # Classify boxes
    main_chars = []      # Area >= 5000 (full characters)
    small_parts = []     # Area < 5000 (diacritics)
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        area = (x2 - x1) * (y2 - y1)
        if area >= MAIN_CHAR_THRESHOLD:
            main_chars.append((i, [x1, y1, x2, y2], area))
        else:
            small_parts.append((i, [x1, y1, x2, y2], area))
    
    print(f"Main characters ({len(main_chars)}): {[i for i, _, _ in main_chars]} areas: {[int(a) for _, _, a in main_chars]}")
    print(f"Small parts ({len(small_parts)}): {[i for i, _, _ in small_parts]} areas: {[int(a) for _, _, a in small_parts]}")
    print()
    
    merged = []
    used = [False] * len(boxes)
    
    # STEP 1: Assign small parts to closest main characters
    part_assignments = {}  # part_idx → best_main_idx (or None)
    
    for idx_j, box_j, area_j in small_parts:
        xj, yj, xj2, yj2 = box_j
        part_center_x = (xj + xj2) / 2
        
        best_main = None
        best_distance = float('inf')
        
        # Find the CLOSEST main character
        for idx_i, box_i, area_i in main_chars:
            x1, y1, x2, y2 = box_i
            char_center_x = (x1 + x2) / 2
            
            # Check if part is above or below (vertically stacked)
            part_above = yj2 < y1
            part_below = yj > y2
            
            if not (part_above or part_below):
                continue
            
            # Calculate distance
            vert_gap = min(abs(yj - y2), abs(yj2 - y1))
            horiz_alignment = abs(part_center_x - char_center_x)
            char_width = x2 - x1
            
            # Check if within merge range
            if horiz_alignment < char_width * 1.0 and vert_gap < 150:
                distance = vert_gap + horiz_alignment
                
                if distance < best_distance:
                    best_distance = distance
                    best_main = idx_i
        
        part_assignments[idx_j] = best_main
    
    # STEP 2: Group small parts that belong together
    small_part_groups = []
    assigned_to_group = set()
    
    for idx_j in [i for i, _, _ in small_parts]:
        if idx_j in assigned_to_group:
            continue
        
        if part_assignments[idx_j] is not None:
            continue
        
        group = [idx_j]
        assigned_to_group.add(idx_j)
        
        xj, yj, xj2, yj2 = [box for i, box, _ in small_parts if i == idx_j][0]
        
        for idx_k in [i for i, _, _ in small_parts]:
            if idx_k in assigned_to_group or part_assignments[idx_k] is not None:
                continue
            
            xk, yk, xk2, yk2 = [box for i, box, _ in small_parts if i == idx_k][0]
            
            horiz_gap = max(0, max(xj, xk) - min(xj2, xk2))
            vert_gap = max(0, max(yj, yk) - min(yj2, yk2))
            distance = np.sqrt(horiz_gap**2 + vert_gap**2)
            
            if distance < 50:
                group.append(idx_k)
                assigned_to_group.add(idx_k)
        
        if group:
            small_part_groups.append(group)
    
    # STEP 3: Group main characters that are close together (like Т pieces)
    main_char_groups = []
    assigned_mains = set()
    
    for idx_i in [i for i, _, _ in main_chars]:
        if idx_i in assigned_mains:
            continue
        
        group = [idx_i]
        assigned_mains.add(idx_i)
        
        x1, y1, x2, y2 = [box for i, box, _ in main_chars if i == idx_i][0]
        
        # Find other main chars close to this one
        for idx_k in [i for i, _, _ in main_chars]:
            if idx_k in assigned_mains:
                continue
            
            xk, yk, xk2, yk2 = [box for i, box, _ in main_chars if i == idx_k][0]
            
            # Calculate distance between main characters
            horiz_gap = max(0, max(x1, xk) - min(x2, xk2))
            vert_gap = max(0, max(y1, yk) - min(y2, yk2))
            distance = np.sqrt(horiz_gap**2 + vert_gap**2)
            
            # If parts of same character are vertically aligned and close (< 40px gap)
            if distance < 40:
                group.append(idx_k)
                assigned_mains.add(idx_k)
        
        if group:
            main_char_groups.append(group)
    
    print(f"Main char groups: {main_char_groups}")
    print(f"Small part groups: {small_part_groups}")
    
    # STEP 4: Merge grouped main characters together FIRST (before individual processing)
    for group in main_char_groups:
        if not group or len(group) < 2:
            continue
        
        # Skip if any in group already used
        if any(used[idx] for idx in group):
            continue
        
        # Start with first main char in group
        idx_first = group[0]
        x1, y1, x2, y2 = main_chars[0][1]  # Get box from first item in main_chars with matching idx
        
        # Find the actual box
        for i, (idx, box, area) in enumerate(main_chars):
            if idx == idx_first:
                x1, y1, x2, y2 = box
                break
        
        # Merge all other main chars in group
        for idx_k in group[1:]:
            # Find the box for this main char
            for i, (idx, box, area) in enumerate(main_chars):
                if idx == idx_k:
                    xk, yk, xk2, yk2 = box
                    x1 = min(x1, xk)
                    y1 = min(y1, yk)
                    x2 = max(x2, xk2)
                    y2 = max(y2, yk2)
                    used[idx_k] = True
                    print(f"Main {idx_k} ({int(area)}px) ✓ grouped with main {idx_first}")
                    break
        
        merged.append([x1, y1, x2, y2])
        used[idx_first] = True
        print(f"✓ Merged group {group} into 1 character")
    
    # STEP 5: Merge small parts with their assigned main characters
    for idx_i, box_i, area_i in main_chars:
        if used[idx_i]:
            continue
        
        x1, y1, x2, y2 = box_i
        
        for idx_j, box_j, area_j in small_parts:
            if used[idx_j]:
                continue
            
            if part_assignments.get(idx_j) != idx_i:
                continue
            
            xj, yj, xj2, yj2 = box_j
            x1 = min(x1, xj)
            y1 = min(y1, yj)
            x2 = max(x2, xj2)
            y2 = max(y2, yj2)
            used[idx_j] = True
            print(f"Part {idx_j} ({int(area_j)}px) ✓ merged with main {idx_i}")
        
        merged.append([x1, y1, x2, y2])
        used[idx_i] = True
    
    # STEP 6: Merge grouped small parts together
    for group in small_part_groups:
        if not group or len(group) < 2:
            continue
        
        if any(used[idx] for idx in group):
            continue
        
        idx_first = group[0]
        x1, y1, x2, y2 = None, None, None, None
        
        # Find boxes for all parts in group
        for i, (idx, box, area) in enumerate(small_parts):
            if idx == idx_first:
                x1, y1, x2, y2 = box
                break
        
        for idx_j in group[1:]:
            for i, (idx, box, area) in enumerate(small_parts):
                if idx == idx_j:
                    xj, yj, xj2, yj2 = box
                    x1 = min(x1, xj)
                    y1 = min(y1, yj)
                    x2 = max(x2, xj2)
                    y2 = max(y2, yj2)
                    used[idx_j] = True
                    print(f"Part {idx_j} ({int(area)}px) ✓ grouped with part {idx_first}")
                    break
        
        merged.append([x1, y1, x2, y2])
        used[idx_first] = True
        print(f"✓ Merged group {group} into 1 character")
    
    # STEP 7: Keep unmerged parts as separate characters
    for idx_j, box_j, area_j in small_parts:
        if not used[idx_j]:
            merged.append(box_j)
            print(f"Kept separate: part {idx_j} ({int(area_j)}px)")
    
    print(f"Final merged: {len(merged)} characters\n")
    return merged

def preprocess_char(char_img):
    """Preprocess character image for model"""
    img = np.where(char_img > 128, 255, 0).astype(np.uint8)
    coords = np.argwhere(img == 255)
    if coords.size == 0:
        return None
    
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape
    
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))
    
    from cv2 import resize
    resized = resize(crop, (new_w, new_h), interpolation=1)
    final = np.zeros((28, 28), np.uint8)
    final[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = resized
    
    tensor = torch.tensor(final/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

# -------------------
# Tkinter App
# -------------------
class MongolianDraw:
    def __init__(self, width=1600, height=900, pen=12):
        self.width = width
        self.height = height
        self.pen = pen
        self.root = tk.Tk()
        self.root.title("Mongolian Character Segmentation")
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = np.zeros((self.height, self.width), np.uint8)
        self.char_boxes = []
        
        # Buttons
        frm = tk.Frame(self.root)
        frm.pack()
        tk.Button(frm, text="Segment", command=self.segment).pack(side=tk.LEFT)
        tk.Button(frm, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(frm, text="Save chars", command=self.save_chars).pack(side=tk.LEFT)
        tk.Button(frm, text="Load Model", command=self.load_model).pack(side=tk.LEFT)
        tk.Button(frm, text="Predict", command=self.predict_all).pack(side=tk.LEFT)
        
        self.model = None
        self.root.mainloop()

    def paint(self, event):
        x, y = event.x, event.y
        for dx in range(-self.pen, self.pen+1):
            for dy in range(-self.pen, self.pen+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.image[ny, nx] = 255
        x1, y1, x2, y2 = x - self.pen, y - self.pen, x + self.pen, y + self.pen
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

    def clear(self):
        self.canvas.delete("all")
        self.image[:] = 0
        self.char_boxes.clear()

    def segment(self):
        self.char_boxes.clear()
        self.canvas.delete("all")
        
        print("=== SEGMENTATION ===")
        
        # Step 1: Find connected components
        components = find_connected_components(self.image, threshold=128)
        print(f"Found {len(components)} components")
        
        # Step 2: Convert to bounding boxes (filters small noise)
        boxes = components_to_boxes(components, min_size=30)
        print(f"After filtering small components: {len(boxes)} boxes")
        
        # Step 3: Merge boxes that are very close (parts of same character)
        self.char_boxes = merge_close_boxes(boxes)
        print(f"After merging close parts: {len(self.char_boxes)} characters")
        print()
        
        # Redraw
        self.canvas.delete("all")
        for y in range(self.height):
            for x in range(self.width):
                if self.image[y, x] > 128:
                    self.canvas.create_line(x, y, x+1, y, fill="gray")
        
        # Draw boxes
        for i, box in enumerate(self.char_boxes):
            x1, y1, x2, y2 = box
            area = (x2-x1) * (y2-y1)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="cyan", width=2)
            self.canvas.create_text(x1+5, y1-10, text=f"{i} ({area}px)", fill="yellow", font=("Arial", 10))

    def save_chars(self):
        if not self.char_boxes:
            return
        folder = "chars_npy"
        os.makedirs(folder, exist_ok=True)
        for i, box in enumerate(self.char_boxes):
            x1, y1, x2, y2 = box
            char_img = self.image[y1:y2, x1:x2]
            np.save(os.path.join(folder, f"char_{i}.npy"), char_img)
        print(f"Saved {len(self.char_boxes)} chars to {folder}")

    def load_model(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if not path:
            return
        self.model = ImprovedCNN()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        print("Model loaded.")

    def predict_all(self):
        if self.model is None or not self.char_boxes:
            return
        for box in self.char_boxes:
            x1, y1, x2, y2 = box
            char_img = self.image[y1:y2, x1:x2]
            tensor = preprocess_char(char_img)
            if tensor is None:
                continue
            with torch.no_grad():
                out = self.model(tensor)
                probs = nn.Softmax(dim=1)(out)
                conf, idx = torch.max(probs, 1)
                label = LABELS.get(idx.item()+1, "?")
                self.canvas.create_text((x1+x2)//2, y1-20, text=f"{label} ({conf.item()*100:.0f}%)",
                                        fill="yellow", font=("Arial", 16, "bold"))

# -------------------
if __name__ == "__main__":
    MongolianDraw(width=1600, height=900, pen=12)