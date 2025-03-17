import os

import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image


pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

from tqdm import tqdm
skull_image_url = '../data/skull/Skull_Infringement/postive'
output_image_url = '../data/skull/diffedit'
skull_images = os.listdir(skull_image_url)
skull_images = sorted(skull_images, key=lambda x: int(os.path.splitext(x)[0]))
# print(skull_images)
# input()
source_prompt = "skull"
target_prompt = "random products"
for skull_image in tqdm(skull_images, desc="Processing images"):
    raw_image = load_image(os.path.join(skull_image_url, skull_image)).resize((768, 768))

    mask_image = pipeline.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))

    inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents

    output_image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        negative_prompt=source_prompt,
    ).images[0]

    # 使用修改后的图像作为新的 raw_image
    raw_image2 = output_image.resize((768, 768))

    # 生成新的 mask 图像
    mask_image = pipeline.generate_mask(
        image=raw_image2,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    # 反向嵌入
    inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image2).latents
    # 生成新的输出图像
    output_image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        negative_prompt=source_prompt,
    ).images[0]

    output_image.save(os.path.join(output_image_url, f"{skull_image}"))
    # mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
    # make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
    