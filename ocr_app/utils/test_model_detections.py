import argparse
from ultralytics import YOLO
import cv2
import time

def detect_and_display(model_name):
    image_path = r"C:\Users\z00511dv\Downloads\test_img3.png"
    image = cv2.imread(image_path)

    model_paths = {
        "yolo": r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\models\yolo.pt",
        "rtdetr": r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\models\rtdetr.pt"
    }

    if model_name not in model_paths:
        print(f"Model '{model_name}' not recognized. Available models: {list(model_paths.keys())}")
        return

    model_path = model_paths[model_name]
    model = YOLO(model_path)

    start_time = time.time()
    results = model(image)
    end_time = time.time()
    print(f"{model_name} inference time: {end_time - start_time:.3f} seconds")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    if hasattr(model, 'names'):
        class_names = [model.names[i] for i in class_ids]
    else:
        class_names = [str(i) for i in class_ids]

    for cls, conf, box in zip(class_names, confidences, boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow(f"Detections by {model_name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection with selected model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--yolo', action='store_true', help="Use YOLO model")
    group.add_argument('--rtdetr', action='store_true', help="Use RTDETR model")

    args = parser.parse_args()

    if args.yolo:
        detect_and_display("yolo")
    elif args.rtdetr:
        detect_and_display("rtdetr")
