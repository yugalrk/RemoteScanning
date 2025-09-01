# SiemensProject

# 🚀 Warning Detection via Augmented Synthetic Images

This project implements a complete pipeline to detect `'steps'` and `'warnings'` from image sequences, focusing on realistic augmentation, compression simulation, and multi-model training.

---

## 📁 Folder Structure

```plaintext
data_no_aug/              # Original raw dataset
data/expanded/            # Output dataset with pasted crops + full augmentations
runs/yolo/                # YOLOv8 training logs and checkpoints
runs/rtdetr/              # RT-DETR logs and results
output/                   # Final trained models (.pt / .pth)
```

---

## 🔄 Augmentation Pipeline

- Extracts warning crops from annotated images (`CLASS_ID_WARNING`)
- Applies **realistic video compression artifacts** via JPEG degradation + posterization
- Pastes crops into step-only or empty images (`CLASS_ID_STEPS`)
- Applies full-image augmentations: blur, noise, dropout, resize
- Saves bounding boxes in YOLO format

### 💡 Tools Used
- `Albumentations v2.0.8` — for transforms (crop_aug, full_aug)
- `OpenCV` — for image loading, artifact simulation
- Custom functions: `simulate_video_artifacts()`, `apply_jpeg_artifact()`, `posterize()`

---

## 🧠 Model Training

### ✅ YOLOv8
- Uses `ultralytics.YOLO("yolov8n.yaml")`
- Trained for 25 epochs with batch size 8
- Learning rate: `5e-4` (custom per model logic)

### ✅ RT-DETR
- Summary: 302 layers, ~32M params
- mAP@50: **0.944**
- mAP@50–95: **0.921**
- Speed: **10.1ms** per image inference

### 🔧 Features
- Auto-extraction of `best.pt`
- Dual-model training loop with conditional learning rates
- GPU usage confirmed: NVIDIA RTX 2000 Ada (CUDA 11.8 + PyTorch 2.7.1)

---

## 📊 Evaluation & Visualization

The pipeline provides mAP scores, precision, recall, inference time.

Plotting scripts (available separately) visualize:
- Detection accuracy (`mAP50`, `mAP5095`)
- Class-wise breakdown (`steps`, `warnings`)
- Model speed comparison

---

## 📝 Requirements

- Python 3.11+
- Ultralytics ≥ 8.3.162
- PyTorch with CUDA support
- OpenCV
- Albumentations 2.x
- Matplotlib

---

## ✨ Highlights

- ⚠️ Robust warning simulation using compression artifacts
- 📦 Smart bounding box recalculation with YOLO-format export
- 📊 Clean metric reporting for YOLOv8 and RT-DETR
- 🔥 GPU-aware training routines
- 🧪 Modular code, ready for extension to other classes or detectors

---
