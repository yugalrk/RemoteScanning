import os
import re
from jiwer import wer

def clean_text(text):
    """Optional cleaning or normalization if needed."""
    return text.strip()

# Paths to folders
extractions_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\extractions_paddle1.5x"
ground_truth_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\ground_truths"

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Get sorted list of filenames
filenames = sorted([
    f for f in os.listdir(ground_truth_folder)
    if f.lower().endswith('.txt')
], key=extract_number)[:10]  # First 10 only

total_score = 0
results = []

for fname in filenames:
    gt_path = os.path.join(ground_truth_folder, fname)
    ocr_fname = fname.replace("extraction_", "")
    ocr_path = os.path.join(extractions_folder, ocr_fname)

    if not os.path.exists(ocr_path):
        print(f"Missing OCR file: {fname}")
        continue

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = clean_text(f.read())

    with open(ocr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ocr_text = clean_text(''.join(lines[:-1]))  # Ignore last line (confidence)

    error = wer(gt_text, ocr_text)
    accuracy = 1.0 - error

    total_score += accuracy
    results.append((fname, accuracy))

# Final report
average_score = total_score / len(results) if results else 0
print(f"\nEvaluated {len(results)} files")
print(f"Average Global WER Accuracy: {average_score:.2%}\n")

for fname, score in results:
    print(f"{fname}: {score:.2%}")
