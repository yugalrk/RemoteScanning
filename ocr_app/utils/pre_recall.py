import os
import re

def clean_text(text):
    """Normalize text by stripping whitespace."""
    return text.strip()

def normalize_text(text):
    """Normalize OCR text by removing duplicate lines and trimming."""
    seen = set()
    processed_lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            processed_lines.append(cleaned)
    return '\n'.join(processed_lines)

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

# Storage for results
total_precision = 0
total_recall = 0
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
        # Ignore last line (confidence), then normalize
        raw_ocr_text = ''.join(lines[:-1])
        ocr_text = normalize_text(raw_ocr_text)

    gt_words = set(gt_text.split())
    ocr_words = set(ocr_text.split())

    true_positives = len(gt_words & ocr_words)
    false_positives = len(ocr_words - gt_words)
    false_negatives = len(gt_words - ocr_words)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    results.append((fname, precision, recall))
    total_precision += precision
    total_recall += recall

# Final report
num_files = len(results)
avg_precision = total_precision / num_files if num_files else 0
avg_recall = total_recall / num_files if num_files else 0

print(f"\nEvaluated {num_files} files")
print(f"Average Precision: {avg_precision:.2%}")
print(f"Average Recall: {avg_recall:.2%}\n")
for fname, precision, recall in results:
    print(f"{fname}: Precision={precision:.2%}, Recall={recall:.2%}")
