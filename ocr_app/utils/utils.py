# app/utils.py
import cv2
import numpy as np

def read_image(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

def crop_region(image, x1, y1, x2, y2):
    return image[int(y1):int(y2), int(x1):int(x2)]
