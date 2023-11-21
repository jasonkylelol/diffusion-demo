from diffusers import DiffusionPipeline 
import torch
from PIL import Image
import gradio
import torch
import datetime


base_model_path = "/app/models/stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_path = "/app/models/stabilityai/stable-diffusion-xl-refiner-1.0"

neg_prompt = "distorted or disproportionate creature, missing or extra limbs, poor anatomy or proportions"

base_pipeln = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# base_pipeln.to("cuda")
base_pipeln.enable_model_cpu_offload()
#base_pipeln.enable_sequential_cpu_offload()
print(f"loading base pretrained from [{base_model_path}]")

refiner_pipeln = DiffusionPipeline.from_pretrained(
    refiner_model_path,
    text_encoder_2 = base_pipeln.text_encoder_2,
    vae = base_pipeln.vae,
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
refiner_pipeln.enable_model_cpu_offload()
#refiner_pipeln.enable_sequential_cpu_offload()
print(f"loading refiner pretrained from [{refiner_model_path}]")


def text_to_image_fn(prompt):
    curtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{curtime}: {prompt}")

    image = base_pipeln(prompt=prompt, negative_prompt=neg_prompt, output_type="latent").images[0]

    image = refiner_pipeln(prompt=prompt, image=image[None, :]).images[0]

    return image


if __name__ == '__main__':
    # Create a Gradio interface with a text input and an image output
    iface = gradio.Interface(fn=text_to_image_fn, inputs="text", outputs="image",
        allow_flagging = "auto",
        title="stable-diffusion-xl Text-to-Image Generator",
        description="Enter a text prompt and see the generated image.")

    # Launch the interface and share it with others
    iface.launch(server_name='0.0.0.0', share=False)

