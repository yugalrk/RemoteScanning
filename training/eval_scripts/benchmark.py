import subprocess

models = ['detr', 'yolov5', 'custom']
for model in models:
    subprocess.run([
        "python", "main.py",
        "--model-name", model,
        "--epochs", "5",
        "--data-path", "data/train",
        "--save-path", f"checkpoints/{model}.pth"
    ])
