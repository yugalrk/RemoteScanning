import os
import shutil
import gc
import yaml
import torch
from ultralytics import RTDETR

def train(args):
    print("🚀 Training RT-DETR")

    # === Step 1: Verify and load dataset config ===
    yaml_path = os.path.join("data", "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ data.yaml missing at {yaml_path}")

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    print(f"📄 Dataset Config | Classes: {data_config['names']} | Train: {data_config['train']} | Val: {data_config['val']}")

    # === Step 2: Load RT-DETR model ===
    model = RTDETR("rtdetr-l.yaml")

    # === Step 3: Train model ===
    results = model.train(
        data='C:/Users/z00511dv/Downloads/DLproj/training/data/data.yaml',
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.lr,
        cache=False,
        amp=True,
        workers=2,
        project=args.log_dir,
        name="exp",
        exist_ok=True
    )

    # === Step 4: Save best model weights ===
    exp_dir = os.path.join(args.log_dir, "exp")
    weight_path = os.path.join(exp_dir, "weights", "best.pt")
    if os.path.exists(weight_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        shutil.copy(weight_path, args.save_path)
        print(f"✅ RT-DETR model saved to {args.save_path}")
    else:
        print("⚠️ Warning: best.pt not found in expected path")

    # === Step 5: Collect metrics ===
    metrics = {
        "epochs": args.epochs,
        "map50": results.box.map50,
        "map": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "training_time_sec": results.speed.get("train", 0),
        "validation_time_sec": results.speed.get("val", 0)
    }

    # === Step 6: Clean up memory ===
    del model
    del results
    gc.collect()
    torch.cuda.empty_cache()

    return metrics
