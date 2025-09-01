import os

# 📁 Set your image folder path
folder_path = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\images'

# 📦 Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# 🔢 Counter for renaming
counter = 1

# 🚀 Loop through and rename
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(image_extensions):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"test_image{counter}{os.path.splitext(filename)[1]}"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")
        counter += 1

print("✅ All images renamed.")
