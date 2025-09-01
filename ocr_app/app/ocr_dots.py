import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from huggingface_hub import snapshot_download

from qwen_vl_utils import vision_process

MODEL_ID = "helizac/dots.ocr-4bit"

local_model_path = snapshot_download(repo_id=MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, use_fast=True)

image_path = r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\output\original\region_1.png"
image = Image.open(image_path)

prompt_text = """\
You are a precise OCR assistant for radiology scheduling/screenshots. Read ONLY the visible text. Do not guess missing text. Preserve the order top-to-bottom.

Task
Extract the table-like list as separate lines. Each line must start with its leading index number. If a line’s starting number is missing or wrong, infer it by sequential continuity from the previous line and correct it.

Output format
Return one line per entry using:
<index> <status> <description> <time>

Rules

index: integer, strictly increasing by 1

status: short tag such as Done, ACT, InRx (copy exactly if legible; else use ?)

description: copy visible text; keep internal hyphens/colons; compress repeated dashes to “—” if needed

time: keep mm:ss if present; if none is visible, leave empty “” (do not invent)

Trim extra spaces; keep normal spaces between tokens

Do not include any content outside the list (headers, toolbar, scrollbars)

Examples of acceptable lines
2 Done 2: Ax T2 FSE fast 00:45
3 Done 3: Brain AirX — 00:22
6 Done 6: Ax T2 FSE HR 01:45
7 Done 7: Ax TOF HS 03:04\
"""

messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.15)

generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output_text)
