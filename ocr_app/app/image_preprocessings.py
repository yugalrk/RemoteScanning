import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola
import imutils
from pathlib import Path

debug_folder = Path(r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\app\debug_outputs')
debug_folder.mkdir(parents=True, exist_ok=True)  # Ensure it exists

def preprocess_for_ocr(crop, window_size=15, k=0.5, debug=False):
    img = np.array(crop) if isinstance(crop, Image.Image) else crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug: Image.fromarray(gray).save(debug_folder/"step1_gray.jpg")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if debug: Image.fromarray(enhanced).save(debug_folder/"step2_clahe.jpg")

    # Gamma correction
    def adjust_gamma(image, gamma=1.2):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    enhanced = adjust_gamma(enhanced)
    if debug: Image.fromarray(enhanced).save(debug_folder/"step3_gamma.jpg")

    # Denoising
    denoise_h = 10
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    if debug: Image.fromarray(denoised).save(debug_folder/"step4_denoised.jpg")

    # 🔍 Contrast check
    # contrast = np.std(denoised)
    # if contrast < 30:
    #     # 🚫 Skip binarization — use contrast stretching
    #     min_val = np.min(denoised)
    #     max_val = np.max(denoised)
    #     stretched = ((denoised - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
    #     binary = stretched
    #     if debug: Image.fromarray(binary).save(debug_folder/"step5_stretched.jpg")
    # else:
    #     # ✅ Apply binarization
    #     try:
    #         sauvola_thresh = threshold_sauvola(denoised, window_size=window_size, k=k)
    #         binary = (denoised > sauvola_thresh).astype(np.uint8) * 255
    #     except Exception:
    #         binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                        cv2.THRESH_BINARY, 15, 10)
    #     if debug: Image.fromarray(binary).save(debug_folder/"step5_binary.jpg")

    # Resize
    rescaled = cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Sharpen
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened = cv2.filter2D(rescaled, -1, kernel)
    if debug: Image.fromarray(sharpened).save(debug_folder/"step5_sharpened.jpg")

    final_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

    return final_img


def preprocess_original(crop):
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
    #os.makedirs("output/preprocessed_original", exist_ok=True)
    #final_img.save(f"output/preprocessed_original/region_{region_id}.png")
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

def preprocess_sauvola(crop, window_size=19, k=0.3, denoise_h=31):

    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2GRAY)
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

    #os.makedirs("output/preprocessed_sauvola", exist_ok=True)
    #final_img.save(f"output/preprocessed_sauvola/region_{region_id}_sauvola.png")
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
