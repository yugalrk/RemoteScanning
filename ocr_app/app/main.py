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

from model import detect_objects  # our detection model
from ocr_qwen import model as ocr_model, processor as ocr_processor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins. For production, restrict origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Output folders outside the app folder
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


def pil_image_to_base64_str(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/detect-and-ocr")
async def detect_and_ocr(
    file: UploadFile = File(...),
    roi_model_name: str = Form("yolo")  # Default to "yolo"
):
    file_bytes = await file.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    boxes, class_names = detect_objects(img_bgr, model_name=roi_model_name)

    ocr_results = []
    start_time = time.time()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img_bgr[y1:y2, x1:x2]

        processed_img, proc_path = preprocess_sauvola(crop, region_id=i+1)

        pil_img = processed_img.convert("RGB") if isinstance(processed_img, Image.Image) else Image.fromarray(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image exactly as it appears including duplicates or near-duplicates. "
                            "Duplicates are perfectly accepted."
                            "Return each detected line separately, one line per output line. "
                            "The text usually starts with a number, make sure the the extracted text has the numbering right and in proper order"
                            "Preserve the original order and formatting. "
                            "Include every visible line exactly as it appears, including duplicates. "
                            "Do not remove or merge repeated lines."
                            "Example lines from the image:"
                            "7: Act 3D Sag T2 FLAIR Cube 03:38"
                            "8: Act 3D Sag T2 FLAIR Cube 05:00"
                            "9: Ax TOF HS 05:00"
                        ),
                    },
                ],
            }
        ]
        prompt = ocr_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = ocr_processor(text=[prompt], images=pil_img, padding=True, return_tensors="pt")
        inputs = inputs.to(ocr_model.device)

        output_ids = ocr_model.generate(**inputs, max_new_tokens=516)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        output_text = ocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Convert processed image to base64 for UI display
        b64_img = pil_image_to_base64_str(processed_img)

        ocr_results.append(
            {
                "region_id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "processed_image_path": proc_path,
                "processed_image_b64": b64_img,
                "extracted_text": output_text,
            }
        )

    end_time = time.time()

    combined_filename = f"ocr_results_{uuid.uuid4()}.txt"
    combined_path = os.path.join(OCR_RESULTS_DIR, combined_filename)
    with open(combined_path, "w", encoding="utf-8") as f:
        for res in ocr_results:
            f.write(f"Region {res['region_id']} bbox {res['bbox']}\n")
            f.write(f"Image: {res['processed_image_path']}\n")
            f.write(f"Text:\n{res['extracted_text']}\n\n{'-'*40}\n")

    return JSONResponse(
        {"ocr_results": ocr_results, "processing_time_sec": round(end_time - start_time, 3), "saved_ocr_file": combined_path}
    )
