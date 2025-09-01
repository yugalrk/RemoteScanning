import os
import yaml
from ultralytics import YOLO

def main():
    # 📍 Step 1: Load your trained model
    model_path = "output/yolo_final_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = YOLO(model_path)

    # 📍 Step 2: Define your original test dataset path
    test_data_path = r"C:\Users\z00511dv\Downloads\DLproj\data_no_aug\test\images"  # Make sure this exists

    # 📍 Step 3: Create custom data.yaml
    original_data_yaml = {
        'train': '',  # No training here
        'val': r"C:\Users\z00511dv\Downloads\DLproj\data_no_aug\train",
        'names': ['steps', 'warnings']  # Adjust if you have more classes
    }

    os.makedirs("data", exist_ok=True)
    yaml_path = "data/original.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(original_data_yaml, f)

    # 📍 Step 4: Run validation
    results = model.val(data=yaml_path, imgsz=640, workers=0)

    # 📍 Step 5: Print detailed metrics
    print("✅ Validation Metrics on Original Test Set")
    print(f"mAP@50:     {results.box.map50:.3f}")
    print(f"mAP@50-95:  {results.box.map:.3f}")
    print(f"Precision:  {results.box.mp:.3f}")
    print(f"Recall:     {results.box.mr:.3f}")

if __name__ == "__main__":
    main()
