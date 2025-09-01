import os
import shutil

label_dir = r"C:\Users\z00511dv\Downloads\DLproj\data\test\labels"
image_dir = r"C:\Users\z00511dv\Downloads\DLproj\data\test\images"
clean_dir = r"C:\Users\z00511dv\Downloads\DLproj\data\test_clean"

os.makedirs(os.path.join(clean_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(clean_dir, "labels"), exist_ok=True)

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) != 1:
        continue  # skip multi-label images

    image_name = os.path.splitext(label_file)[0] + ".jpg"  # or .png if needed
    image_path = os.path.join(image_dir, image_name)

    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(clean_dir, "images", image_name))
        shutil.copy(label_path, os.path.join(clean_dir, "labels", label_file))

print("✅ Clean test set created with single-label images only.")
