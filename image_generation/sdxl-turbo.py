from diffusers import AutoPipelineForText2Image
import torch
import os
from PIL import Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

def save_image(image, image_name):
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_file = os.path.join(output_dir, image_name)
    image.save(image_file, format="JPEG")
    
    return image_file

save_image(image, 'test.jpg')