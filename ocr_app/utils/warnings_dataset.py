import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Paths
IMAGE_FOLDER = r'C:\Users\z00511dv\Downloads\DLproj\training\data_no_aug\train\images'
LABEL_FOLDER = r'C:\Users\z00511dv\Downloads\DLproj\training\data_no_aug\train\labels'
OUTPUT_FOLDER = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\warnings'
CLASS_NAME = "warning"
CLASS_ID = 1
MAX_WARNINGS = 20
SIMILARITY_THRESHOLD = 0.95  # SSIM ranges from 0 to 1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

saved_crops = []

def is_similar(new_crop, saved_crops):
    new_arr = np.array(new_crop.resize((128, 128))).astype("float32") / 255.0
    for old_crop in saved_crops:
        old_arr = np.array(old_crop.resize((128, 128))).astype("float32") / 255.0
        score = ssim(new_arr, old_arr, data_range=1.0, channel_axis=2)
        if score > SIMILARITY_THRESHOLD:
            return True
    return False


warning_count = 0

for filename in os.listdir(LABEL_FOLDER):
    if not filename.endswith('.txt') or warning_count >= MAX_WARNINGS:
        continue

    label_path = os.path.join(LABEL_FOLDER, filename)
    image_name = filename.replace('.txt', '.jpg')
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if not os.path.exists(image_path):
        continue

    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if warning_count >= MAX_WARNINGS:
                break

            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, cx, cy, w, h = map(float, parts)
            if int(cls_id) != CLASS_ID:
                continue

            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            cropped = img.crop((x1, y1, x2, y2))

            if is_similar(cropped, saved_crops):
                continue

            saved_crops.append(cropped)
            output_name = f"{image_name.replace('.jpg', '')}_{CLASS_NAME}_{warning_count}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            cropped.save(output_path)

            warning_count += 1

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")

print(f"✅ Saved {warning_count} unique '{CLASS_NAME}' regions to {OUTPUT_FOLDER}")
