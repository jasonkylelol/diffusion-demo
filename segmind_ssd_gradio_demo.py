from diffusers import AutoPipelineForText2Image 
import torch
from PIL import Image
import gradio
import torch
import datetime

model_path = "/root/huggingface/models/segmind/SSD-1B/"
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16",
    )
# pipeline.enable_model_cpu_offload()
pipeline.to("cuda:2")

neg_prompt = "distorted or disproportionate creature, missing or extra limbs, poor biological anatomy or proportions"

def text_to_image_fn(prompt):
    curtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{curtime}: {prompt}")
    image = pipeline(prompt=prompt, negative_prompt=neg_prompt, num_inference_steps=125).images[0]
    return image


if __name__ == '__main__':
    # Create a Gradio interface with a text input and an image output
    iface = gradio.Interface(fn=text_to_image_fn, inputs="text", outputs="image",
        allow_flagging = "auto",
        title="segmind/SSD-1B Text-to-Image Generator",
        description="Enter a text prompt and see the generated image.")

    # Launch the interface and share it with others
    iface.launch(server_name='0.0.0.0', share=False)

