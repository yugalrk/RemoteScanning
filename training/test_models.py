from ultralytics import YOLO
import cv2
import time
import os

def detect_and_save(model_name):
    # Hardcoded input image path
    image_path = r"C:\Users\z00511dv\OneDrive - Siemens Healthineers\Desktop\screenshot_20250613_171600_0005_png.rf.c40badd21e671fe671a322db1ea40f7e.jpg"
    image = cv2.imread(image_path)

    model_paths = {
        "yolo": r"C:\Users\z00511dv\Downloads\DLproj\training\runs\yolo\exp\weights\best.pt",
        "rtdetr": r"C:\Users\z00511dv\Downloads\DLproj\training\runs\rtdetr\exp\weights\best.pt"
    }

    if model_name not in model_paths:
        print(f"Model '{model_name}' not recognized. Available models: {list(model_paths.keys())}")
        return

    model_path = model_paths[model_name]
    model = YOLO(model_path)
    print("model classes", model.names)
    start_time = time.time()
    results = model(image, conf=0.50)
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
        # Draw bounding boxes and labels on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Create output folder if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the annotated image
    output_path = os.path.join(output_dir, f"detections_{model_name}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved output image to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run detection and save result")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--yolo', action='store_true', help="Use YOLO model")
    group.add_argument('--rtdetr', action='store_true', help="Use RTDETR model")

    args = parser.parse_args()

    if args.yolo:
        detect_and_save("yolo")
    elif args.rtdetr:
        detect_and_save("rtdetr")
