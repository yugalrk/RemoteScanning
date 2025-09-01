import os
from PIL import Image
import imagehash

# 🔧 Folder containing images
IMAGE_FOLDER = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset'
HASH_THRESHOLD = 5  # Lower = stricter similarity

# 📦 Track hashes and duplicates
hashes = {}
duplicates = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path)
        img_hash = imagehash.phash(img)

        # Compare with existing hashes
        for existing_hash, existing_file in hashes.items():
            if abs(img_hash - existing_hash) <= HASH_THRESHOLD:
                duplicates.append(path)
                break
        else:
            hashes[img_hash] = filename

# 🗑️ Delete duplicates
for dup in duplicates:
    os.remove(dup)
    print(f"Deleted: {dup}")

print(f"✅ Removed {len(duplicates)} similar images.")
