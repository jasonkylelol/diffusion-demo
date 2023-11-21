from diffusers import AutoPipelineForImage2Image
from PIL import Image
import torch

#model_path = "runwayml/stable-diffusion-v1-5"
model_path = "/app/models/runwayml/stable-diffusion-v1-5/"

pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"


image = Image.open("input/1665_Girl_with_a_Pearl_Earring.jpg").convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_steps=200, strength=0.80, guidance_scale=10.5).images[0]

save_file = "output/dog_wearing_pearl_earring.jpg"
print(f"{type(image)} save to {save_file}")
image.save(save_file)

