from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


# default_bnb_config = BitsAndBytesConfig()
# print("load_in_4bit:", default_bnb_config.load_in_4bit)
# print("load_in_8bit:", default_bnb_config.load_in_8bit)
# print("bnb_4bit_use_double_quant:", getattr(default_bnb_config, "bnb_4bit_use_double_quant", None))
# print("bnb_4bit_quant_type:", getattr(default_bnb_config, "bnb_4bit_quant_type", None))
# print("bnb_8bit_use_double_quant:", getattr(default_bnb_config, "bnb_8bit_use_double_quant", None))
# print("bnb_8bit_quant_type:", getattr(default_bnb_config, "bnb_8bit_quant_type", None))

#Default configs are computationally expensive and slow, so using custom configs
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,   # Optional: improves accuracy/performance
    bnb_8bit_quant_type="nf8"         # Normal float quantization, good quality
)

processor = AutoProcessor.from_pretrained("JackChew/Qwen2-VL-2B-OCR")
model = AutoModelForImageTextToText.from_pretrained("JackChew/Qwen2-VL-2B-OCR", quantization_config=bnb_config, device_map="auto")
print("Qwen-VL-loaded succesfully")