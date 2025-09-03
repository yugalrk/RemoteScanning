from paddleocr import PaddleOCR
import cv2
import numpy as np
#from ocr_app.app.image_preprocessings import preprocess_for_ocr

image_path = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\images\test_image8.png"
img = cv2.imread(image_path)
print(img.shape)
#img = preprocess_for_ocr(img)
img_small = cv2.resize(img, None, fx=1.5, fy=1.5)

if img_small is None:
    print("Image not loaded, check path.")
else:
    print("initialting OCR")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    try:
        results = ocr.predict(img_small)
        print("OCR call finished")

        page_result = results[0]
        rec_texts = page_result.get('rec_texts', [])
        rec_polys = page_result.get('rec_polys', [])
        rec_scores = page_result.get('rec_scores', [])

        # Confidence filtering threshold
        confidence_threshold = 0.75

        # Group text by line (vertical proximity)
        lines = []
        line_threshold = 10

        for i, box in enumerate(rec_polys):
            score = rec_scores[i]
            if score < confidence_threshold:
                continue  # skip low-confidence texts

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

        # Sort and join texts per line
        def sort_key(item):
            box, txt = item
            x_coords = [point[0] for point in box]
            return min(x_coords)

        output_lines = []
        all_scores = [score for score in rec_scores if score >= confidence_threshold]

        for line in lines:
            line['texts'].sort(key=sort_key)
            line_text = " ".join(txt for _, txt in line['texts'])
            output_lines.append(line_text)

        print("\n".join(output_lines))

        # Compute and print average confidence score over filtered texts
        if all_scores:
            avg_confidence = sum(all_scores) / len(all_scores)
            print(f"\nAverage confidence score (filtered): {avg_confidence:.2%}")
        else:
            print("\nNo confidence scores passing threshold.")

    except Exception as e:
        print("Error during OCR:", e)
