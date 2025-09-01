from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# 🖼️ Load your test image
image_path = r"C:\Users\z00511dv\Downloads\image.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 📦 Load your models
model_paths = [
    "output/rtdetr.pt",
    "output/yolo.pt",
   #"output/detr_final.pt"
]

models = [YOLO(path) for path in model_paths]

# 🔍 Run predictions
results = [model.predict(img_rgb, conf=0.05, verbose=False)[0] for model in models]
print(len(results[0].boxes))
# 🎨 Visualize predictions side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, result, path in zip(axes, results, model_paths):
    ax.imshow(result.plot())
    ax.set_title(f"Predictions: {os.path.basename(path)}")
    ax.axis("off")

plt.tight_layout()
plt.show()
