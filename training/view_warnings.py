import os
import cv2
import matplotlib.pyplot as plt

# === CONFIGURATION ===
IMG_DIR = "data/expanded/train/images"
LBL_DIR = "data/expanded/train/labels"
CLASS_ID_WARNING = "1"
EXT = ".jpg"
NUM_SAMPLES = 4  # Only show 4 images

def load_labels(label_path):
    if not os.path.exists(label_path): return []
    with open(label_path) as f:
        return [line.strip().split() for line in f if line.strip()]

# === FIND IMAGES THAT CONTAIN WARNINGS ===
import random

all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(EXT)]
random.shuffle(all_imgs)  # Shuffle full list

warning_imgs = []
for fname in all_imgs:
    lbl_path = os.path.join(LBL_DIR, fname.replace(EXT, ".txt"))
    labels = load_labels(lbl_path)
    if any(lbl[0] == CLASS_ID_WARNING for lbl in labels):
        warning_imgs.append(fname)
    if len(warning_imgs) >= NUM_SAMPLES:
        break


# === VISUALIZE ===
fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(16, 6))
if NUM_SAMPLES == 1:
    axes = [axes]

for ax, fname in zip(axes, warning_imgs):
    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(EXT, ".txt"))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    labels = load_labels(lbl_path)

    for lbl in labels:
        if lbl[0] != CLASS_ID_WARNING: continue
        _, x, y, bw, bh = map(float, lbl)
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, "warning", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    ax.imshow(img)
    ax.set_title(fname, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
