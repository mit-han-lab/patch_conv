import argparse

import torch
from diffusers import AutoencoderKL
from patch_conv import convert_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, nargs="*", default=4096, help="Image size of generation")

    args = parser.parse_args()
    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
    else:
        if len(args.image_size) == 1:
            args.image_size = [args.image_size[0], args.image_size[0]]
        else:
            assert len(args.image_size) == 2
    args.image_size = [args.image_size[0] // 8, args.image_size[1] // 8]

    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=repo_id, subfolder="vae", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    convert_model(vae)
    dtype = next(vae.parameters()).dtype
    latent = torch.randn(1, 4, *args.image_size, dtype=dtype).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        vae.decode(latent)
    print(f"Max used memory: {torch.cuda.max_memory_allocated() / 1024**3} G")
