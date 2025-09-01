import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import time
from skimage.filters import threshold_sauvola, threshold_niblack

# 🧠 Initialize Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\z00511dv\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

def preprocess_original(crop, region_id):
    # Original preprocessing method
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.fastNlMeansDenoising(thresh, None, h=30, templateWindowSize=7, searchWindowSize=21)
    rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(rescaled, -1, kernel)
    final_img = Image.fromarray(sharpened)
    os.makedirs("output/preprocessed_original", exist_ok=True)
    final_img.save(f"output/preprocessed_original/region_{region_id}.png")
    return final_img

def preprocess_otsu(crop, region_id, denoise_h=30):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(otsu_thresh, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(rescaled, -1, kernel)
    final_img = Image.fromarray(sharpened)
    os.makedirs("output/preprocessed_otsu", exist_ok=True)
    final_img.save(f"output/preprocessed_otsu/region_{region_id}_otsu.png")
    return final_img

def preprocess_sauvola(crop, region_id, window_size=19, k=0.3, denoise_h=31):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sauvola_thresh = threshold_sauvola(enhanced, window_size=window_size, k=k)
    sauvola_binary = (enhanced > sauvola_thresh).astype(np.uint8) * 255
    denoised = cv2.fastNlMeansDenoising(sauvola_binary, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(rescaled, -1, kernel)

    # # Define a structuring element - try small kernel like 2x2 or 3x3
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    # # Apply dilation (or closing)
    # processed = cv2.dilate(sharpened, kernel, iterations=1)

    # Update final_img with processed array
    final_img = Image.fromarray(sharpened)

    os.makedirs("output/preprocessed_sauvola", exist_ok=True)
    final_img.save(f"output/preprocessed_sauvola/region_{region_id}_sauvola.png")
    return final_img

def preprocess_niblack(crop, region_id, window_size=25, k=0.1, denoise_h=29):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    niblack_thresh = threshold_niblack(gray, window_size=window_size, k=k)
    niblack_binary = (gray > niblack_thresh).astype(np.uint8) * 255
    denoised = cv2.fastNlMeansDenoising(niblack_binary, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    rescaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(rescaled, -1, kernel)
    final_img = Image.fromarray(sharpened)
    os.makedirs("output/preprocessed_niblack", exist_ok=True)
    final_img.save(f"output/preprocessed_niblack/region_{region_id}_niblack.png")
    return final_img

def run_ocr_on_regions(image, boxes):
    os.makedirs("output/original", exist_ok=True)
    output_texts = []
    ocr_total_time = 0.0

    # Define whitelist and blacklist characters
    whitelist_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-,:"
    blacklist_chars = "!@#$%^&*()«_+=[]{}|\\_;<>/?`~"  # example characters to exclude

    # Combine into Tesseract config
    custom_config = (
        f'--oem 3 --psm 6 '
        f'-c tessedit_char_whitelist={whitelist_chars} '
        f'-c tessedit_char_blacklist={blacklist_chars}'
    )

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = image[y1:y2, x1:x2]

        # Save original crop
        original_path = f"output/original/region_{i+1}.png"
        cv2.imwrite(original_path, crop)

        print(f"Processing Region {i+1}...")

        # Use the same custom_config here, no reassignment
        start_ocr = time.time()
        original_img = preprocess_original(crop, i+1)
        text_original = pytesseract.image_to_string(original_img, config=custom_config)
        end_ocr = time.time()
        ocr_time_original = end_ocr - start_ocr

        start_ocr = time.time()
        otsu_img = preprocess_otsu(crop, i+1)
        text_otsu = pytesseract.image_to_string(otsu_img, config=custom_config)
        end_ocr = time.time()
        ocr_time_otsu = end_ocr - start_ocr

        start_ocr = time.time()
        sauvola_img = preprocess_sauvola(crop, i+1)
        text_sauvola = pytesseract.image_to_string(sauvola_img, config=custom_config)
        end_ocr = time.time()
        ocr_time_sauvola = end_ocr - start_ocr


        start_ocr = time.time()
        niblack_img = preprocess_niblack(crop, i+1)
        text_niblack = pytesseract.image_to_string(niblack_img, config=custom_config)
        end_ocr = time.time()
        ocr_time_niblack = end_ocr - start_ocr

        ocr_total_time += (ocr_time_original + ocr_time_otsu + ocr_time_sauvola+ ocr_time_niblack)

        output_texts.append(
            f"Region {i+1} OCR Results:\n"
            f"Original \n ({ocr_time_original:.3f}s):\n{text_original.strip()}\n \n"
            f"Otsu \n({ocr_time_otsu:.3f}s):\n{text_otsu.strip()}\n \n"
            f"Sauvola \n({ocr_time_sauvola:.3f}s):\n{text_sauvola.strip()}\n \n"
            #f"niblack ({ocr_time_niblack:.3f}s):\n{text_niblack.strip()}\n"

            + "-" * 40
        )

        print(output_texts[-1])

    return output_texts, ocr_total_time
