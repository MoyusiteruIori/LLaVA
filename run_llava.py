from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

device = "cuda:1"

model_path = "/data/mengyu/huggingface/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    use_flash_attn=False,
    device_map=device,
    mm_vision_tower="/data/mengyu/huggingface/clip-vit-large-patch14-336",
    attn_implementation="eager"
)
model: LlavaLlamaForCausalLM
conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()

prompt = f"{DEFAULT_IMAGE_TOKEN}\nWhat is the name of the game?"
image = Image.open("samples/img2.jpg").convert("RGB")
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(device)

conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
print(input_ids.shape)

outputs = model.generate(
    inputs=input_ids,
    images=image_tensor,
    do_sample=False,
    max_new_tokens=1024,
    use_cache=True,
    return_dict_in_generate=True,
)

print(tokenizer.decode(outputs["sequences"][0], skip_special_tokens=False).strip())
