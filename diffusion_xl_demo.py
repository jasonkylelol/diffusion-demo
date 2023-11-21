from diffusers import DiffusionPipeline
import torch
import os

base_model_path = "/app/models/stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_path = "/app/models/stabilityai/stable-diffusion-xl-refiner-1.0"
output_dir = "/app/tutorials/output"

def init_sd_xl_pipeline():
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

    return base_pipeln, refiner_pipeln

if __name__ == '__main__':
    base_pipeln, refiner_pipeln = init_sd_xl_pipeline()
    
    prompt = "a white pomeranian wearing red baseball cap and standing on desert"
    #prompt = "an astronaut riding a white horse"
    save_file = os.path.join(output_dir, "diffusion_xl_output.jpg")

    #image = pipeln(prompt, num_inference_steps=115, height=768, width=1024, guidance_scale=8.5).images[0]
    image = base_pipeln(prompt=prompt, output_type="latent").images[0]

    image = refiner_pipeln(prompt=prompt, image=image[None, :]).images[0]

    print(f"{type(image)} save to {save_file}")
    image.save(save_file)

