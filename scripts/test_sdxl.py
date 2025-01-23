import os
import random

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import trange

from patch_conv import convert_model

if __name__ == "__main__":
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipeline.to("cuda")
    pipeline.vae = convert_model(pipeline.vae, splits=4)

    prompt = "Beautiful landscape, snow, mountains, glaciers, vivid colors."
    pipeline.set_progress_bar_config(position=1, desc="images", leave=False)
    for i in trange(8, position=0, desc="Generating images", leave=False):
        seed = random.randint(0, 100000000)
        with torch.no_grad():
            image = pipeline(
                width=3840,
                height=3840,
                prompt=prompt,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                guidance_scale=8,
                num_inference_steps=50,
            ).images[0]

            torch.cuda.empty_cache()
            print("Max used memory: ", torch.cuda.max_memory_allocated() / 1024**3)
        # print(image[0].shape)
        os.makedirs("sdxl_images", exist_ok=True)
        image.save(f"sdxl_images/{seed}.png")
