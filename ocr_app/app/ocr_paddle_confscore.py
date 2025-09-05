from paddleocr import PaddleOCR
import cv2
import numpy as np

image_path = r"C:\Users\z00511dv\Downloads\screenshot_20250613_171651_0012 (1).png"
img = cv2.imread(image_path)
print(img.shape)

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

        # Group text by line (vertical proximity)
        lines = []
        line_threshold = 30

        for i, box in enumerate(rec_polys):
            text = rec_texts[i]
            y_coords = [point[1] for point in box]
            y_center = np.mean(y_coords)

            placed = False
            for line in lines:
                if abs(line['y_center'] - y_center) < line_threshold:
                    line['texts'].append((box, text, rec_scores[i]))
                    line['y_center'] = (line['y_center'] * (len(line['texts']) - 1) + y_center) / len(line['texts'])
                    placed = True
                    break
            if not placed:
                lines.append({'y_center': y_center, 'texts': [(box, text, rec_scores[i])]})

        # Sort and join texts per line, including confidence beside text
        def sort_key(item):
            box, txt, score = item
            x_coords = [point[0] for point in box]
            return min(x_coords)

        output_lines = []
        all_scores = []
        for line in lines:
            line['texts'].sort(key=sort_key)
            # Format each text segment with confidence in parentheses
            formatted_texts = [f"{txt}({score:.2f})" for _, txt, score in line['texts']]
            line_text = " ".join(formatted_texts)
            output_lines.append(line_text)
            all_scores.extend([score for _, _, score in line['texts']])

        print("\n".join(output_lines))

        # Compute and print average confidence score
        if all_scores:
            avg_confidence = sum(all_scores) / len(all_scores)
            print(f"\nAverage confidence score: {avg_confidence:.2%}")
        else:
            print("\nNo confidence scores found.")

    except Exception as e:
        print("Error during OCR:", e)
