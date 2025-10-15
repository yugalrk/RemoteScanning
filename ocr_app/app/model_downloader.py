import os
import requests

paddlex_dir = r"C:\Users\z00511dv\.paddlex\official_models"

model_files = {
    "en_PP-OCRv5_mobile_rec": [
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/en_PP-OCRv5_mobile_rec/inference.pdmodel",
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/en_PP-OCRv5_mobile_rec/inference.pdiparams",
    ],
    "PP-LCNet_x1_0_doc_ori": [
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-LCNet_x1_0_doc_ori/inference.pdmodel",
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-LCNet_x1_0_doc_ori/inference.pdiparams",
    ],
    "PP-OCRv5_server_det": [
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-OCRv5_server_det/inference.pdmodel",
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-OCRv5_server_det/inference.pdiparams",
    ],
    "PP-OCRv5_server_rec": [
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-OCRv5_server_rec/inference.pdmodel",
        "https://downloads.sourceforge.net/project/paddlex.mirror/v3.2.1/PP-OCRv5_server_rec/inference.pdiparams",
    ],
}

def download_file(url, file_path):
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping...")
        return

    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, verify=False)  # Disable SSL verification for now
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved {file_path}")
    else:
        print(f"Failed to download {url} with status {r.status_code}")

for model, urls in model_files.items():
    model_dir = os.path.join(paddlex_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    for url in urls:
        filename = url.split("/")[-1]  # Correct filename extraction
        file_path = os.path.join(model_dir, filename)
        download_file(url, file_path)

print("Model files download completed.")
