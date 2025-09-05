import os
import re

def tokenize(text):
    return re.findall(r'\b[A-Za-z0-9]+\b', text.lower())

def line_tokenize(text):
    """Split text to list of token lists by line."""
    lines = text.strip().split('\n')
    return [tokenize(line) for line in lines]

def line_match_score(gt_tokens, ocr_tokens):
    """Compute match rate between two token lists."""
    gt_set = set(gt_tokens)
    ocr_set = set(ocr_tokens)
    if not gt_set:
        return 1.0 if not ocr_set else 0.0
    matched = gt_set & ocr_set
    return len(matched) / len(gt_set)

def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0
    intersect = set1 & set2
    union = set1 | set2
    return len(intersect) / len(union)

def positional_aware_score_strict(ocr_text, gt_text, window=1, similarity_threshold=0.7):
    gt_lines = line_tokenize(gt_text)
    ocr_lines = line_tokenize(ocr_text)
    
    matched_ocr_indices = set()
    scores = []
    
    for i, gt_line in enumerate(gt_lines):
        best_score = 0
        best_j = None
        
        for j in range(max(0, i - window), min(len(ocr_lines), i + window + 1)):
            if j in matched_ocr_indices:
                continue  # OCR line already matched
            
            sim = jaccard_similarity(gt_line, ocr_lines[j])
            if sim >= similarity_threshold:
                score = line_match_score(gt_line, ocr_lines[j])
                if score > best_score:
                    best_score = score
                    best_j = j
        
        if best_j is not None:
            matched_ocr_indices.add(best_j)
            scores.append(best_score)
        else:
            scores.append(0)  # No suitable match found for this line
    
    return sum(scores) / len(scores) if scores else 0

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
    # OCR filename may differ (remove "extraction_" prefix)
    ocr_fname = fname.replace("extraction_", "")
    ocr_path = os.path.join(extractions_folder, ocr_fname)

    if not os.path.exists(ocr_path):
        print(f"Missing OCR file: {fname}")
        continue

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read()

    with open(ocr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # ignore last line (confidence)
        ocr_text = ''.join(lines[:-1])

    result_score = positional_aware_score_strict(ocr_text, gt_text)
    total_score += result_score
    results.append((fname, result_score))

# Final report
average_score = total_score / len(results) if results else 0
print(f"\n🔍 Evaluated {len(results)} files")
print(f"📊 Average Match Rate: {average_score:.2%}\n")

for fname, score in results:
    print(f"{fname}: {score:.2%}")
