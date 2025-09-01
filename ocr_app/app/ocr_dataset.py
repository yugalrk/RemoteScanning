import os
import random
from PIL import Image

from model import detect_objects

# 🔧 Paths
INPUT_FOLDER = r'C:\Users\z00511dv\Downloads\DLproj\training\data_no_aug\train\images'
OUTPUT_FOLDER = 'ocr_app/ocr_dataset'
MODEL_NAME = 'rtdetr'

# 📦 Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 🖼️ Get image files and shuffle
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)

# 🚀 Crop and save up to 30 regions
crop_count = 0
for img_name in image_files:
    if crop_count >= 30:
        break

    img_path = os.path.join(INPUT_FOLDER, img_name)
    image = Image.open(img_path).convert('RGB')

    boxes, _ = detect_objects(image, model_name=MODEL_NAME)

    for box in boxes:
        if crop_count >= 30:
            break
        x1, y1, x2, y2 = map(int, box[:4])  # Ensure it's just the coordinates
        cropped = image.crop((x1, y1, x2, y2))
        crop_name = f"{os.path.splitext(img_name)[0]}_crop_{crop_count}.png"
        cropped.save(os.path.join(OUTPUT_FOLDER, crop_name))
        crop_count += 1

print(f"✅ Saved {crop_count} cropped regions to {OUTPUT_FOLDER}")
