import argparse
import os
import json
from train_yolo import train as train_yolo
from train_rtdetr import train as train_rtdetr
import multiprocessing


multiprocessing.set_start_method('spawn', force=True)


def get_args():
    parser = argparse.ArgumentParser(description="Unified RT-DETR + YOLO Training")
    parser.add_argument("--models", type=str, nargs="+", choices=["yolo", "rtdetr"],
                        default=["yolo", "rtdetr"], help="Models to train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,  # Allow dynamic override
                        help="Initial learning rate (optional, model-specific if not set)")
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--metrics-json", type=str, default="output/training_metrics.json")
    return parser.parse_args()

def main():

    args = get_args()
    all_metrics = {}

    os.makedirs(args.save_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\n🚀 Training model: {model_name.upper()}")

        if args.lr is None:
            args.lr = 5e-4 if model_name == "yolo" else 2e-5

        model_save_path = os.path.join(args.save_dir, f"{model_name}_final_model.pth")
        model_log_dir  = os.path.join(args.log_dir, model_name)

        # Attach paths to args
        args.save_path = model_save_path
        args.log_dir = model_log_dir

        # Train model
        if model_name == "yolo":
            metrics = train_yolo(args)
        elif model_name == "rtdetr":
            metrics = train_rtdetr(args)
        else:
            print(f"⚠️ Unknown model: {model_name}")
            continue

        # Collect performance metrics for plotting
        all_metrics[model_name] = metrics

    # Save metrics to JSON
    with open(args.metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n📈 Training metrics saved to: {args.metrics_json}")

if __name__ == "__main__":
    main()
