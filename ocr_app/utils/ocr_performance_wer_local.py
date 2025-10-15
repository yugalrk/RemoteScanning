import os
import re
from jiwer import wer

def tokenize(text):
    return re.findall(r'\b[A-Za-z0-9]+\b', text.lower())

def line_tokenize(text):
    """Split text to list of lines (strings)."""
    return text.strip().split('\n')

def positional_aware_wer(ocr_text, gt_text, window=1):
    gt_lines = line_tokenize(gt_text)
    ocr_lines = line_tokenize(ocr_text)

    matched_ocr_indices = set()
    scores = []

    for i, gt_line in enumerate(gt_lines):
        best_wer = 1.0  # Worst possible WER (100% errors)
        best_j = None

        # Look for best matching OCR line within window
        for j in range(max(0, i - window), min(len(ocr_lines), i + window + 1)):
            if j in matched_ocr_indices:
                continue  # OCR line already matched
            
            current_wer = wer(gt_line, ocr_lines[j])
            if current_wer < best_wer:
                best_wer = current_wer
                best_j = j

        if best_j is not None:
            matched_ocr_indices.add(best_j)
            scores.append(best_wer)
        else:
            scores.append(1.0)  # No match found, count as max WER

    return 1.0 - (sum(scores) / len(scores)) if scores else 0.0  # Convert to accuracy-like score

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
        gt_text = f.read()

    with open(ocr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ocr_text = ''.join(lines[:-1])  # Ignore last line if confidence

    score = positional_aware_wer(ocr_text, gt_text)
    total_score += score
    results.append((fname, score))

# Final report
average_score = total_score / len(results) if results else 0
print(f"\nEvaluated {len(results)} files")
print(f"Average Positional-Aware Line-Wise WER Accuracy: {average_score:.2%}\n")

for fname, score in results:
    print(f"{fname}: {score:.2%}")
