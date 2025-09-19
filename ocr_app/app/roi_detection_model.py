from ultralytics import YOLO

# Load your trained models once
model1 = YOLO(r"C:\\Users\\z00511dv\\Downloads\\DLproj\\ocr_app\\models\\yolo_best.pt")
model2 = YOLO(r"C:\\Users\\z00511dv\\Downloads\\DLproj\\ocr_app\\models\\rtdetr_best.pt")

# Map model names to model instances
ROI_MODELS = {
    "yolo": model1,
    "rtdetr": model2,
}

def detect_objects(image, model_name="yolo"):
    model = ROI_MODELS.get(model_name)
    if model is None:
        raise ValueError(f"ROI model '{model_name}' not found.")

    results = model(image, conf=0.50)

    # If results is a list, take the first element
    if isinstance(results, list):
        results = results[0]

    # Extract bounding boxes, class indices, and confidence scores
    boxes = results.boxes.data.cpu().numpy()  # shape: (n, 6) with [x1, y1, x2, y2, conf, cls]
    class_indices = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    # Map class indices to class names
    class_names = results.names

    # Prepare a list of detection details for output
    detections = []
    for box, conf, cls_idx in zip(boxes, confidences, class_indices):
        x1, y1, x2, y2 = map(int, box[:4])
        class_name = class_names[int(cls_idx)]
        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": float(conf),
            "class": class_name
        })

    return detections
