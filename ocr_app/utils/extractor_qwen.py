from ocr_qwen import model, processor
from PIL import Image
import time


if __name__ == '__main__':
    # Initialize the loader (will automatically use GPU if available)
    ocr_model = model

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                            "Extract all text from this image exactly as it appears including duplicates or near-duplicates. "
                            "Duplicates are perfectly accepted."
                            "Return each detected line separately, one line per output line. "
                            "The text usually starts with a number, make sure the the extracted text has the numbering right and in proper order"
                            "Preserve the original order and formatting. "
                            "Include every visible line exactly as it appears, including duplicates. "
                            "Do not remove or merge repeated lines."
                            "Example lines from the image:"
                            "7: Act 3D Sag T2 FLAIR Cube 03:38"
                            "8: Act 3D Sag T2 FLAIR Cube 05:00"
                            "9: Ax TOF HS 05:00"

                    )
                }
            ]
        }
    ]


    # Specify your input image path
    image_path =  r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\output\preprocessed_sauvola\region_1_sauvola.png"  # <-- change to your image filename
    img = Image.open(image_path)
    w,h = img.size
    new_size = (w//2,h//2)
    resized_img = img.resize(new_size)
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    start = time.time()
    inputs = processor(text=[text_prompt], images=resized_img, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=516)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    end = time.time()
    structured_lines = output_text[0].split('\n')
    for line in structured_lines:
        print(line.strip())
    print(f"time taken for extraction: {end-start}")
