import os
import uuid
import time
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from image_preprocessings import preprocess_for_ocr
from roi_detection_model import detect_objects  # your detection model
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

# You can customize the base directory here before server startup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Customize your base final output folder here
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "final_output")
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

ocr_wrapper = PaddleOcrWrapper()  # Your modular OCR


def pil_image_to_base64_str(pil_img: Image.Image):
    from io import BytesIO
    import base64
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    base64_str = b64_bytes.decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


# @app.post("/detect-and-ocr")
# async def detect_and_ocr(
#     file: UploadFile = File(...),
#     roi_model_name: str = Form("yolo"),
#     use_preprocessing: bool = Form(True),
# ):
#     file_bytes = await file.read()
#     np_img = np.frombuffer(file_bytes, np.uint8)
#     img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     if img_bgr is None:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Could not decode image. Please upload a valid image file."},
#         )

#     # Create a new session folder with timestamp under final output
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     session_dir = os.path.join(FINAL_OUTPUT_DIR, f"session_{timestamp}")
#     os.makedirs(session_dir, exist_ok=True)

#     # Create subfolders for input images, steps, warnings, and extracted texts
#     input_images_dir = os.path.join(session_dir, "input_images")
#     detected_steps_dir = os.path.join(session_dir, "detected_steps")
#     detected_warnings_dir = os.path.join(session_dir, "detected_warnings")
#     extracted_text_steps_dir = os.path.join(session_dir, "extracted_text_steps")
#     extracted_text_warnings_dir = os.path.join(session_dir, "extracted_text_warnings")

#     for directory in [
#         input_images_dir,
#         detected_steps_dir,
#         detected_warnings_dir,
#         extracted_text_steps_dir,
#         extracted_text_warnings_dir,
#     ]:
#         os.makedirs(directory, exist_ok=True)

#     # Save input screenshot
#     input_image_path = os.path.join(input_images_dir, f"screenshot_{timestamp}.png")
#     cv2.imwrite(input_image_path, img_bgr)

#     detections = detect_objects(img_bgr, model_name=roi_model_name)

#     ocr_results = []
#     start_time = time.time()

#     for i, det in enumerate(detections):
#         x1, y1, x2, y2 = det["box"]
#         img_h, img_w = img_bgr.shape[:2]
#         x1 = max(0, min(img_w, x1))
#         y1 = max(0, min(img_h, y1))
#         x2 = max(0, min(img_w, x2))
#         y2 = max(0, min(img_h, y2))

#         crop = img_bgr[y1:y2, x1:x2]
#         class_name = det["class"].lower()
#         confidence = det["confidence"]

#         if use_preprocessing:
#             preprocessed_img = preprocess_for_ocr(crop)
#         else:
#             preprocessed_img = crop

#         if not isinstance(preprocessed_img, Image.Image):
#             pil_img = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
#         else:
#             pil_img = preprocessed_img
#         start = time.time()
#         output_lines, avg_confidence = ocr_wrapper.predict(pil_img)
#         end = time.time()
#         print(f"OCr time for detections:{end-start}")
#         output_text = "\n".join(output_lines)

#         # Save crops and text based on class
#         if class_name == "steps":
#             crop_path = os.path.join(detected_steps_dir, f"step_{i+1}_{timestamp}.png")
#             text_path = os.path.join(extracted_text_steps_dir, f"step_{i+1}_{timestamp}.txt")
#         elif class_name in ["warning", "warnings", "warningss"]:
#             crop_path = os.path.join(detected_warnings_dir, f"warning_{i+1}_{timestamp}.png")
#             text_path = os.path.join(extracted_text_warnings_dir, f"warning_{i+1}_{timestamp}.txt")
#         else:
#             # Optionally handle unknown classes
#             crop_path = os.path.join(session_dir, f"{class_name}_{i+1}_{timestamp}.png")
#             text_path = os.path.join(session_dir, f"{class_name}_{i+1}_{timestamp}.txt")

#         cv2.imwrite(crop_path, crop)
#         with open(text_path, "w", encoding="utf-8") as f:
#             f.write(output_text)

#         b64_img = pil_image_to_base64_str(pil_img)

#         ocr_results.append(
#             {
#                 "region_id": i + 1,
#                 "bbox": [x1, y1, x2, y2],
#                 "class": class_name,
#                 "confidence": confidence,
#                 "processed_image_path": crop_path,
#                 "processed_image_b64": b64_img,
#                 "extracted_text": output_text,
#                 "average_confidence": avg_confidence,
#             }
#         )

#     end_time = time.time()

#     return JSONResponse(
#         {
#             "ocr_results": ocr_results,
#             "processing_time_sec": round(end_time - start_time, 3),
#             "saved_input_screenshot": input_image_path,
#         }
#     )
@app.post("/detect-and-ocr")
async def detect_and_ocr(
    file: UploadFile = File(...),
    roi_model_name: str = Form("yolo"),
    use_preprocessing: bool = Form(True),
):
    file_bytes = await file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    
    start_total = time.time()
    start_decode = time.time()
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    end_decode = time.time()
    print(f"Time for image decode: {end_decode - start_decode:.4f}s")

    if img_bgr is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image. Please upload a valid image file."},
        )

    start_detect = time.time()
    detections = detect_objects(img_bgr, model_name=roi_model_name)
    end_detect = time.time()
    print(f"Time for detection: {end_detect - start_detect:.4f}s")

    ocr_results = []
    start_ocr_total = time.time()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        img_h, img_w = img_bgr.shape[:2]
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        crop = img_bgr[y1:y2, x1:x2]
        class_name = det["class"].lower()
        confidence = det["confidence"]

        if use_preprocessing:
            start_preproc = time.time()
            preprocessed_img = preprocess_for_ocr(crop)
            end_preproc = time.time()
            print(f"Preprocessing time for crop {i+1}: {end_preproc - start_preproc:.4f}s")
        else:
            preprocessed_img = crop

        if not isinstance(preprocessed_img, Image.Image):
            pil_img = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = preprocessed_img

        start_ocr = time.time()
        output_lines, avg_confidence = ocr_wrapper.predict(pil_img)
        end_ocr = time.time()
        print(f"OCR time for crop {i+1}: {end_ocr - start_ocr:.4f}s")

        output_text = "\n".join(output_lines)

        # Commented out saving for performance benchmark
        # if class_name == "steps":
        #     crop_path = os.path.join(detected_steps_dir, f"step_{i+1}_{timestamp}.png")
        #     text_path = os.path.join(extracted_text_steps_dir, f"step_{i+1}_{timestamp}.txt")
        # elif class_name in ["warning", "warnings", "warningss"]:
        #     crop_path = os.path.join(detected_warnings_dir, f"warning_{i+1}_{timestamp}.png")
        #     text_path = os.path.join(extracted_text_warnings_dir, f"warning_{i+1}_{timestamp}.txt")
        # else:
        #     crop_path = os.path.join(session_dir, f"{class_name}_{i+1}_{timestamp}.png")
        #     text_path = os.path.join(session_dir, f"{class_name}_{i+1}_{timestamp}.txt")

        # cv2.imwrite(crop_path, crop)
        # with open(text_path, "w", encoding="utf-8") as f:
        #     f.write(output_text)

        b64_img = pil_image_to_base64_str(pil_img)

        ocr_results.append(
            {
                "region_id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence,
                "processed_image_path": None,  # No files saved now
                "processed_image_b64": b64_img,
                "extracted_text": output_text,
                "average_confidence": avg_confidence,
            }
        )

    end_ocr_total = time.time()
    print(f"Total OCR time for all crops: {end_ocr_total - start_ocr_total:.4f}s")

    end_total = time.time()
    print(f"Total pipeline time: {end_total - start_total:.4f}s")

    return JSONResponse(
        {
            "ocr_results": ocr_results,
            "processing_time_sec": round(end_total - start_total, 3),
            "saved_input_screenshot": None,
        }
    )
