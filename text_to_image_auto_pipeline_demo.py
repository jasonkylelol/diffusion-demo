from diffusers import AutoPipelineForText2Image 
import torch
from PIL import Image

#model_path = "runwayml/stable-diffusion-v1-5"
model_path = "/app/models/runwayml/stable-diffusion-v1-5/"

prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

save_file = "output/diffusion_v1_5_output.png"

pipeline = AutoPipelineForText2Image.from_pretrained(
    model_path, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None,
).to("cuda")

#image = pipeline(prompt, num_inference_steps=225, height=768, width=1024, guidance_scale=8.5).images[0]
image = pipeline(prompt, num_inference_steps=125).images[0]
print(f"{type(image)} save to {save_file}")
image.save(save_file)

