# main.py
import os
import uuid
import time
import numpy as np
import cv2
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

from model import detect_objects  # your detection model
from ocr_paddle import PaddleOcrWrapper  # Import OCR wrapper class

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PREPROCESSED_ROIS_DIR = os.path.join(OUTPUT_DIR, "preprocessed_rois")
OCR_RESULTS_DIR = os.path.join(OUTPUT_DIR, "ocr_results")

os.makedirs(PREPROCESSED_ROIS_DIR, exist_ok=True)
os.makedirs(OCR_RESULTS_DIR, exist_ok=True)


def preprocess_sauvola(crop_bgr, region_id, window_size=19, k=0.3, denoise_h=31):
    from skimage.filters import threshold_sauvola
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sauvola_thresh = threshold_sauvola(enhanced, window_size=window_size, k=k)
    sauvola_binary = (enhanced > sauvola_thresh).astype(np.uint8) * 255
    denoised = cv2.fastNlMeansDenoising(sauvola_binary, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(rescaled, -1, kernel)
    final_img = Image.fromarray(sharpened)
    save_path = os.path.join(PREPROCESSED_ROIS_DIR, f"region_{region_id}_sauvola.png")
    final_img.save(save_path)
    return final_img, save_path


def pil_image_to_base64_str(pil_img: Image.Image):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    base64_str = b64_bytes.decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


ocr_wrapper = PaddleOcrWrapper()


@app.post("/detect-and-ocr")
async def detect_and_ocr(file: UploadFile = File(...), roi_model_name: str = Form("yolo")):
    file_bytes = await file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    boxes, class_names = detect_objects(img_bgr, model_name=roi_model_name)

    ocr_results = []
    start_time = time.time()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img_bgr[y1:y2, x1:x2]

        processed_img, proc_path = preprocess_sauvola(crop, region_id=i + 1)

        pil_img = (
            processed_img.convert("RGB") if isinstance(processed_img, Image.Image) else Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        )

        output_lines, avg_confidence = ocr_wrapper.predict(pil_img)
        output_text = "\n".join(output_lines)

        b64_img = pil_image_to_base64_str(processed_img)

        ocr_results.append(
            {
                "region_id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "processed_image_path": proc_path,
                "processed_image_b64": b64_img,
                "extracted_text": output_text,
                "average_confidence": avg_confidence,
            }
        )

    end_time = time.time()

    combined_filename = f"ocr_results_{uuid.uuid4()}.txt"
    combined_path = os.path.join(OCR_RESULTS_DIR, combined_filename)
    with open(combined_path, "w", encoding="utf-8") as f:
        for res in ocr_results:
            f.write(f"Region {res['region_id']} bbox {res['bbox']}\n")
            f.write(f"Image: {res['processed_image_path']}\n")
            f.write(f"Text:\n{res['extracted_text']}\n")
            f.write(f"Avg Confidence: {res['average_confidence']:.2%}\n\n{'-'*40}\n")

    return JSONResponse(
        {
            "ocr_results": ocr_results,
            "processing_time_sec": round(end_time - start_time, 3),
            "saved_ocr_file": combined_path,
        }
    )
