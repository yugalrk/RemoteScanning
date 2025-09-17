import os
import re

def tokenize(text):
    return re.findall(r'\b[A-Za-z0-9]+\b', text.lower())

def compare_extraction(ocr_text, ground_truth):
    ocr_tokens = set(tokenize(ocr_text))
    gt_tokens = set(tokenize(ground_truth))

    matched = ocr_tokens & gt_tokens
    missed = gt_tokens - ocr_tokens
    extra = ocr_tokens - gt_tokens

    match_rate = len(matched) / max(len(gt_tokens), 1)

    return {
        "match_rate": round(match_rate, 4),
        "matched_tokens": matched,
        "missed_tokens": missed,
        "extra_tokens": extra
    }

# Paths to folders
extractions_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\warnings\extractions"
ground_truth_folder = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\warnings\ground_truths"

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Get sorted list of filenames
filenames = sorted([
    f for f in os.listdir(ground_truth_folder)
    if f.lower().endswith('.txt')
], key=extract_number)[:20]  # First 10 only

total_score = 0
results = []

for fname in filenames:
    gt_path = os.path.join(ground_truth_folder, fname)
    #ocr_path = os.path.join(extractions_folder, fname)
    #ocr_fname = fname.replace("extraction_", "")
    ocr_path = os.path.join(extractions_folder, gt_path)


    if not os.path.exists(ocr_path):
        print(f"Missing OCR file: {fname}")
        continue

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.readlines()
        gt_text = ''.join(gt_text[:-1])

    with open(ocr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ocr_text = ''.join(lines[:-1])

    result = compare_extraction(ocr_text, gt_text)
    total_score += result['match_rate']
    results.append((fname, result['match_rate']))

# Final report
average_score = total_score / len(results) if results else 0
print(f"\n🔍 Evaluated {len(results)} files")
print(f"📊 Average Match Rate: {average_score:.2%}\n")

for fname, score in results:
    print(f"{fname}: {score:.2%}")
