from diffusers import AutoPipelineForText2Image
import torch
import gradio
import datetime


model_path = "/root/huggingface/models/stabilityai/sdxl-turbo"
device = "cuda:2"


pipe = AutoPipelineForText2Image.from_pretrained(model_path,
    torch_dtype=torch.float16,
    variant="fp16")
pipe.to(device)


def text_to_image_fn(prompt):
    curtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{curtime}: {prompt}")

    image = pipe(prompt=prompt, num_inference_steps=10, guidance_scale=0.0).images[0]
    return image


if __name__ == '__main__':
    # Create a Gradio interface with a text input and an image output
    iface = gradio.Interface(fn=text_to_image_fn, inputs="text", outputs="image",
        allow_flagging = "auto",
        title="sdxl-turbo Text-to-Image Generator",
        description="Enter a text prompt and see the generated image.")

    # Launch the interface and share it with others
    iface.launch(server_name='0.0.0.0', share=True)

