from PaddleOCR.paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image

class PaddleOcrWrapper:
    def __init__(self, confidence_threshold=0.75, line_threshold=30):
        import paddle
        print("using device:",paddle.get_device())
        print("compiled with cuda:",paddle.is_compiled_with_cuda())
        self.ocr = PaddleOCR(
            # text_detection_model_dir=r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\paddle_models\PP-OCRv5_server_det",
            # text_recognition_model_dir=r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\paddle_models\en_PP-OCRv5_mobile_rec",
            # textline_orientation_model_dir=r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\paddle_models\PP-LCNet_x1_0_textline_ori",
            use_textline_orientation=True,
            lang='en'
        )
        self.confidence_threshold = confidence_threshold
        self.line_threshold = line_threshold

    def predict(self, pil_img: Image.Image):
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # Optional: make resizing configurable or conditional
        img_np = cv2.resize(img_np, None, fx=1.5, fy=1.5)

        results = self.ocr.ocr(img_np)
        if not results:
            return [], 0.0

        page_result = results[0]
        rec_texts = page_result.get('rec_texts', [])
        rec_polys = page_result.get('rec_polys', [])
        rec_scores = page_result.get('rec_scores', [])

        lines = []
        for i, box in enumerate(rec_polys):
            score = rec_scores[i]
            if score < self.confidence_threshold:
                continue

            text = rec_texts[i]
            y_coords = [point[1] for point in box]
            y_center = np.mean(y_coords)

            placed = False
            for line in lines:
                if abs(line['y_center'] - y_center) < self.line_threshold:
                    line['texts'].append((box, text))
                    line['y_center'] = (line['y_center'] * (len(line['texts']) - 1) + y_center) / len(line['texts'])
                    placed = True
                    break
            if not placed:
                lines.append({'y_center': y_center, 'texts': [(box, text)]})

        def sort_key(item):
            box, _ = item
            x_coords = [point[0] for point in box]
            return min(x_coords)

        output_lines = []
        all_scores = [score for score in rec_scores if score >= self.confidence_threshold]

        for line in lines:
            line['texts'].sort(key=sort_key)
            line_text = " ".join(txt for _, txt in line['texts'])
            output_lines.append(line_text)

        avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return output_lines, avg_confidence



# Optional: instantiate globally if needed for direct call
ocr_model = PaddleOcrWrapper()
