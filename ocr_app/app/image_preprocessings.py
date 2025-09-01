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
