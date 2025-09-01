from ultralytics import YOLO
import cv2
import time
import os
#from ocr_easy import run_ocr_on_regions
from ocr_tess import run_ocr_on_regions

# 📸 Load image
#image_path = r"C:\Users\z00511dv\Downloads\Screenshot 2025-08-14 152537.png"
#image_path = r"C:\Users\z00511dv\Downloads\Screenshot 2025-08-14 152537.png"
image_path = r"C:\Users\z00511dv\Downloads\test_img3.png"
#image_path = r"C:\Users\z00511dv\Downloads\Screenshot 2025-08-14 151527.png"
#image_path = r"C:\Users\z00511dv\Downloads\Screenshot 2025-08-14 151527.png"

image = cv2.imread(image_path)

# 🧠 Load YOLO model
model = YOLO(r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\models\yolo.pt")

# ⏱️ Run YOLO
start_yolo = time.time()
results = model(image)
end_yolo = time.time()
yolo_time = end_yolo - start_yolo

# 📦 Extract boxes
boxes = results[0].boxes.xyxy.cpu().numpy()
print("✅ YOLO detection complete")
print("📦 Bounding boxes:", boxes)
print(f"📦 Detected {len(boxes)} objects")
print(f"⏱️ YOLO inference time: {yolo_time:.3f} seconds")

# 🔍 Run OCR
if boxes.any():
    output_texts, ocr_total_time = run_ocr_on_regions(image, boxes)

    # 💾 Save results
    os.makedirs("output", exist_ok=True)
    with open("output/ocr_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_texts))

    print("📁 OCR results saved to output/ocr_results.txt")
    print(f"⏱️ Total OCR time: {ocr_total_time:.3f} seconds")
else:
    print("⚠️ No detections found. Skipping OCR.")
