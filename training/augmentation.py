# === YOUR FINAL WARNING AUGMENTATION PIPELINE ===
import os
import cv2
import random
import albumentations as A
import numpy as np

# === CONFIGURATION ===
SRC_ROOT = "data_no_aug"
OUT_ROOT = "data/expanded"
SPLITS = ["train", "valid"]
IMG_EXT = ".jpg"
CLASS_ID_WARNING = "1"
CLASS_ID_STEPS = "0"
NUM_AUG = 3
SCALES = [1.0, 1.2, 1.4]
MAX_CROPS_PER_IMG = 3

TARGET_RATIO_MAP = {
    "train": 0.5,
    "valid": 0.3
}

# === ARTIFACT SIMULATION ===
def apply_jpeg_artifact(img, quality=35):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, enc = cv2.imencode('.jpg', img, params)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if success else img

def simulate_video_artifacts(img):
    img = apply_jpeg_artifact(img, quality=random.randint(30, 40))
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img

# === AUGMENTATION PIPELINES ===
crop_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.4),
    A.MotionBlur(blur_limit=3, p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.Downscale(scale_min=0.5, scale_max=0.7, p=0.3),
])

post_paste_aug = A.Compose([
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2)
])

full_aug = A.Compose([
    A.ISONoise(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.MultiplicativeNoise(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.RandomGamma(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.Resize(height=720, width=1280)
])

# === UTILS ===
def load_labels(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [line.strip().split() for line in f if line.strip()]

def write_labels(path, labels):
    with open(path, "w") as f:
        for label in labels:
            f.write(" ".join(label) + "\n")

def yolo_to_pixel(box, w, h):
    _, x, y, ww, hh = map(float, box)
    x1 = int((x - ww / 2) * w)
    y1 = int((y - hh / 2) * h)
    x2 = int((x + ww / 2) * w)
    y2 = int((y + hh / 2) * h)
    return x1, y1, x2, y2

def boxes_overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

def alpha_blend(bg, crop, px, py):
    bh, bw = bg.shape[:2]
    ch, cw = crop.shape[:2]

    # Clip if it overflows image boundaries
    if py + ch > bh:
        ch = bh - py
    if px + cw > bw:
        cw = bw - px

    crop = crop[:ch, :cw]
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        return bg

    # Ensure proper alpha shape (h, w, 1)
    mask = np.ones((ch, cw), dtype=np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3) / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # Make it (h, w, 1)

    crop = crop.astype(np.float32)
    roi = bg[py:py + ch, px:px + cw].astype(np.float32)

    blended = roi * (1 - alpha) + crop * alpha
    bg[py:py + ch, px:px + cw] = blended.astype(np.uint8)
    return bg



# === MAIN AUGMENTATION LOOP ===
for split in SPLITS:
    print(f"\n🔁 Processing split: {split}")
    src_img_dir = os.path.join(SRC_ROOT, split, "images")
    src_lbl_dir = os.path.join(SRC_ROOT, split, "labels")
    out_img_dir = os.path.join(OUT_ROOT, split, "images")
    out_lbl_dir = os.path.join(OUT_ROOT, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # === Extract warning crops ===
    cutouts = []
    for fname in os.listdir(src_img_dir):
        if not fname.endswith(IMG_EXT): continue
        img = cv2.imread(os.path.join(src_img_dir, fname))
        h, w = img.shape[:2]
        labels = load_labels(os.path.join(src_lbl_dir, fname.replace(IMG_EXT, ".txt")))

        for box in labels:
            if box[0] == CLASS_ID_WARNING:
                x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    for _ in range(3):
                        artifacted = simulate_video_artifacts(crop)
                        aug_crop = crop_aug(image=artifacted)["image"]
                        if aug_crop.size > 0:
                            cutouts.append(aug_crop)

    print(f"✅ Extracted {len(cutouts)} warning crops")

    img_files = [f for f in os.listdir(src_img_dir) if f.endswith(IMG_EXT)]
    target_count = int(TARGET_RATIO_MAP.get(split, 0.25) * len(img_files))
    count_added = 0

    for fname in img_files:
        img = cv2.imread(os.path.join(src_img_dir, fname))
        h, w = img.shape[:2]
        lbl_path = os.path.join(src_lbl_dir, fname.replace(IMG_EXT, ".txt"))
        labels = load_labels(lbl_path)
        class_ids = {lbl[0] for lbl in labels}

        boxes_xyxy = [yolo_to_pixel(lbl, w, h) for lbl in labels]
        n_inserted = 0

        if CLASS_ID_WARNING not in class_ids and cutouts and count_added < target_count:
            for _ in range(MAX_CROPS_PER_IMG):
                crop = random.choice(cutouts)
                scale = random.choice(SCALES)
                crop = cv2.resize(crop, None, fx=scale, fy=scale)
                ch, cw = crop.shape[:2]
                if ch > h or cw > w: continue

                for _ in range(20):  # Try 20 positions to find non-overlapping one
                    px = random.randint(0, w - cw)
                    py = random.randint(0, h - ch)
                    new_box = (px, py, px + cw, py + ch)
                    if all(not boxes_overlap(new_box, b) for b in boxes_xyxy):
                        crop = post_paste_aug(image=crop)["image"]
                        img = alpha_blend(img, crop, px, py)
                        xc = (px + cw / 2) / w
                        yc = (py + ch / 2) / h
                        wn = cw / w
                        hn = ch / h
                        labels.append([CLASS_ID_WARNING, f"{xc:.6f}", f"{yc:.6f}", f"{wn:.6f}", f"{hn:.6f}"])
                        boxes_xyxy.append(new_box)
                        n_inserted += 1
                        break

        if n_inserted:
            count_added += 1

        cv2.imwrite(os.path.join(out_img_dir, fname), img)
        write_labels(os.path.join(out_lbl_dir, fname.replace(IMG_EXT, ".txt")), labels)

    print(f"📈 Warnings added: {count_added} / Target: {target_count}")

    # === Full image augmentations ===
    base_files = [f for f in os.listdir(out_img_dir) if f.endswith(IMG_EXT) and "_aug" not in f]
    for fname in base_files:
        img = cv2.imread(os.path.join(out_img_dir, fname))
        label_path = os.path.join(out_lbl_dir, fname.replace(IMG_EXT, ".txt"))
        if not os.path.exists(label_path): continue
        label_data = open(label_path).read()
        base_name = os.path.splitext(fname)[0]

        for i in range(NUM_AUG):
            aug = full_aug(image=img)
            aug_img = aug["image"]
            aug_fname = f"{base_name}_aug{i+1}{IMG_EXT}"
            cv2.imwrite(os.path.join(out_img_dir, aug_fname), aug_img)
            with open(os.path.join(out_lbl_dir, aug_fname.replace(IMG_EXT, ".txt")), "w") as f:
                f.write(label_data)

    print(f"🎨 Applied {NUM_AUG} augmentations per image for split: {split}")
