from ultralytics import YOLO, RTDETR
import time
import yaml
import os
import json
from pathlib import Path


def evaluate_model(model, name, data_yaml):
    print(f"\n🔍 Evaluating {name}...")
    start = time.time()

    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=0,
        plots=False,
        verbose=False
    )

    duration = time.time() - start
    speed_ms = metrics.speed['inference']
    fps = 1000 / speed_ms if speed_ms > 0 else 0

    print(f"\n📊 Results for {name}:")
    print(f"  mAP@0.5:        {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95:   {metrics.box.map:.3f}")
    print(f"  Precision:      {metrics.box.mp:.3f}")
    print(f"  Recall:         {metrics.box.mr:.3f}")
    print(f"  Inference Time: {duration:.2f} s total | ~{speed_ms:.2f} ms/img")
    print(f"  Estimated FPS:  {fps:.1f} images/sec")

    # Extract per-class mAP@0.5
    per_class_map = {}
    class_names = list(metrics.names.values())  # Ensure it's a list of names

    for i, class_name in enumerate(class_names):
        try:
            class_map = metrics.box.maps[i]
            per_class_map[class_name] = round(class_map, 3) if class_map is not None else 0.0
        except IndexError:
            per_class_map[class_name] = 0.0  # Class missing from metrics


    return {
        "name": name,
        "map50": metrics.box.map50,
        "map": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "inference_time": duration,
        "fps": fps,
        "per_class_map50": per_class_map
    }


def main():
    data_yaml = r"C:\Users\z00511dv\Downloads\DLproj\data\data.yaml"
    output_dir = Path(r"C:\Users\z00511dv\Downloads\DLproj\model performance")
    output_dir.mkdir(exist_ok=True)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"❌ data.yaml not found at {data_yaml}")

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)
    print(f"📄 Dataset Config | Classes: {config['names']} | Train: {config['train']} | Val/Test: {config['val']}")

    # Load models
    yolo_model = YOLO("output/yolo.pt")
    rtdetr_model = RTDETR("output/rtdetr.pt")

    # Run evaluation
    yolo_results = evaluate_model(yolo_model, "YOLOv8", data_yaml)
    rtdetr_results = evaluate_model(rtdetr_model, "RT-DETR", data_yaml)

    # Save results
    results = {
        "YOLOv8": yolo_results,
        "RT-DETR": rtdetr_results
    }

    with open(output_dir / "performance_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n📁 Metrics saved to: {output_dir / 'performance_metrics.json'}")

    # Comparison Table
    print("\n📈 Model Comparison Summary:")
    print(f"{'Metric':<20}{'YOLOv8':>10}{'RT-DETR':>10}")
    print("-" * 40)
    for metric in ["map50", "map", "precision", "recall", "fps"]:
        print(f"{metric:<20}{yolo_results[metric]:>10.3f}{rtdetr_results[metric]:>10.3f}")
    print(f"{'Inference Time (s)':<20}{yolo_results['inference_time']:>10.2f}{rtdetr_results['inference_time']:>10.2f}")

if __name__ == "__main__":
    main()
