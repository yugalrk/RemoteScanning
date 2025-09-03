from ultralytics import YOLO

# Load your trained models once
model1 = YOLO(r"C:\\Users\\z00511dv\\Downloads\\DLproj\\ocr_app\\models\\yolo.pt")
model2 = YOLO(r"C:\\Users\\z00511dv\\Downloads\\DLproj\\ocr_app\\models\\rtdetr.pt")

# Map model names to model instances
ROI_MODELS = {
    "yolo": model1,
    "rtdetr": model2,
}

def detect_objects(image, model_name="yolo"):
    model = ROI_MODELS.get(model_name)
    if model is None:
        raise ValueError(f"ROI model '{model_name}' not found.")

    results = model(image)

    # If results is a list, take the first element
    if isinstance(results, list):
        results = results[0]

    boxes = results.boxes.data.cpu().numpy()
    class_names = results.names
    return boxes, class_names
