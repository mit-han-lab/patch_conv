import os
import random

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import trange

from patch_conv import convert_model

if __name__ == "__main__":
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline.vae = convert_model(pipeline.vae, splits=4)

    prompt = "Beautiful landscape, snow, mountains, glaciers, vivid colors."
    pipeline.set_progress_bar_config(position=1, desc="images", leave=False)
    for i in trange(1, position=0, desc="Generating images", leave=False):
        seed = i
        with torch.no_grad():
            image = pipeline(
                width=4096,
                height=4096,
                prompt=prompt,
                generator=torch.Generator().manual_seed(seed),
                guidance_scale=4.5,
                num_inference_steps=40,
            ).images[0]

            torch.cuda.empty_cache()
            print("Max used memory: ", torch.cuda.max_memory_allocated() / 1024**3)
        os.makedirs("sd3_images", exist_ok=True)
        image.save(f"sd3_images/{seed}_.png")
