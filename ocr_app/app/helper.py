import os
import requests
from pathlib import Path

def download_file(url, target_path):
    if target_path.exists():
        print(f"{target_path.name} already exists, skipping.")
        return
    print(f"Downloading {target_path.name}...")
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {target_path}")

def main():
    base_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english"
    model_files = [
        "en_PP-OCRv3_det_infer.tar",
        "en_PP-OCRv3_rec_infer.tar",
    ]

    cache_dir = Path.home() / ".paddleocr" / "inference" / "en"
    os.makedirs(cache_dir, exist_ok=True)

    for file_name in model_files:
        url = f"{base_url}/{file_name}"
        target_path = cache_dir / file_name
        download_file(url, target_path)

if __name__ == "__main__":
    main()
