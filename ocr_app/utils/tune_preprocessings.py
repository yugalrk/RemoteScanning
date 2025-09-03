import os
from pathlib import Path
from image_preprocessings import preprocess_for_ocr
from PIL import Image
from skimage.measure import shannon_entropy

def process_images_with_paths():
    # 🔧 Define your input and output folders
    input_folder = Path(r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\images')
    output_folder = Path(r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\app\imgs')

    # ✅ Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # 📂 Get first 10 image files (sorted for consistency)
    image_files = sorted([
        f for f in input_folder.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])[:1]

    # 🌀 Process each image
    for idx, image_path in enumerate(image_files):
        try:
            print(f"🔍 Processing: {image_path.name}")
            img = Image.open(image_path)
            entropy = shannon_entropy(img)
            print(entropy)
            processed = preprocess_for_ocr(img, debug=True)
            # 💾 Save processed image
            save_path = output_folder / f"{image_path.stem}_processed{image_path.suffix}"
            processed.save(save_path)
            print(f"✅ Saved to: {save_path}")
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")

# 🚀 Run the function
process_images_with_paths()
