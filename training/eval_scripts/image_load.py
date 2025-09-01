import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to your image folder
image_folder = "data/train/images"

# List all image files (you can filter for .jpg, .png, etc.)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Load and display the first few images
for i, img_name in enumerate(image_files[:5]):  # Display first 5 images
    img_path = os.path.join(image_folder, img_name)
    image = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Image: {img_name}")
    plt.axis('off')
    plt.show()
