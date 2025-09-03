import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Paths
input_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\images"
output_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\extractions_paddle1.5x"
os.makedirs(output_folder, exist_ok=True)

# Initialize OCR once
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Confidence filtering threshold
confidence_threshold = 0.75
line_threshold = 10

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Image not loaded: {filename}")
        continue

    print(f"\nProcessing: {filename}")
    img_small = cv2.resize(img, None, fx=1.5, fy=1.5)

    try:
        results = ocr.predict(img_small)
        page_result = results[0]
        rec_texts = page_result.get('rec_texts', [])
        rec_polys = page_result.get('rec_polys', [])
        rec_scores = page_result.get('rec_scores', [])

        lines = []
        all_scores = []

        for i, box in enumerate(rec_polys):
            score = rec_scores[i]
            if score < confidence_threshold:
                continue

            text = rec_texts[i]
            y_coords = [point[1] for point in box]
            y_center = np.mean(y_coords)

            placed = False
            for line in lines:
                if abs(line['y_center'] - y_center) < line_threshold:
                    line['texts'].append((box, text))
                    line['y_center'] = (line['y_center'] * (len(line['texts']) - 1) + y_center) / len(line['texts'])
                    placed = True
                    break
            if not placed:
                lines.append({'y_center': y_center, 'texts': [(box, text)]})

            all_scores.append(score)

        def sort_key(item):
            box, txt = item
            x_coords = [point[0] for point in box]
            return min(x_coords)

        output_lines = []
        for line in lines:
            line['texts'].sort(key=sort_key)
            line_text = " ".join(txt for _, txt in line['texts'])
            output_lines.append(line_text)

        # Save to file
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
            if all_scores:
                avg_conf = sum(all_scores) / len(all_scores)
                f.write(f"\n\nAverage confidence score (filtered): {avg_conf:.2%}")
            else:
                f.write("\n\nNo confidence scores passing threshold.")

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error during OCR for {filename}: {e}")