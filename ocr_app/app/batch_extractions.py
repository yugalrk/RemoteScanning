from ocr_qwen import model, processor
from PIL import Image
import os
from image_preprocessings import preprocess_sauvola, preprocess_original
# 📁 Folder containing images
image_folder = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\images'
output_folder = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\ground_truths'
output_folder_og = r'C:\Users\z00511dv\Downloads\DLproj\ocr_app\ocr_dataset\ground_truths_og'


# 🧠 OCR prompt template
conversation_template = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Extract all text from this image exactly as it appears including duplicates or near-duplicates. "
                    "Duplicates are perfectly accepted. "
                    "Return each detected line separately, one line per output line. "
                    "The text usually starts with a number, make sure the extracted text has the numbering right and in proper order. "
                    "Preserve the original order and formatting. "
                    "Include every visible line exactly as it appears, including duplicates. "
                    "Do not remove or merge repeated lines. "
                    "Example lines from the image: "
                    "7: Act 3D Sag T2 FLAIR Cube 03:38 "
                    "8: Act 3D Sag T2 FLAIR Cube 05:00 "
                    "9: Ax TOF HS 05:00"
                )
            }
        ]
    }
]

# 🚀 Process each image
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
        image_path = os.path.join(image_folder, filename)

        # Load and resize image
        img = Image.open(image_path)
        w, h = img.size
        resized_img = img.resize((w , h))
        resized_img = preprocess_sauvola(resized_img)

        # Prepare prompt
        text_prompt = processor.apply_chat_template(conversation_template, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=resized_img, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Run OCR
        output_ids = model.generate(**inputs, max_new_tokens=516)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Save output
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder_og, f"extraction_{base_name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_text[0].split('\n'):
                f.write(line.strip() + '\n')
