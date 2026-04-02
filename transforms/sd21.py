import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionImg2ImgPipeline
from PIL import Image
import os


def load_pipeline(model_id="flax/stable-diffusion-2-1"):
    pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        safety_checker=None,
        feature_extractor=None,
    )
    return pipeline, params


def img2img(pipeline, params, image_path: str, prompt: str, strength: float = 0.75, num_inference_steps: int = 50) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, upper = (w - side) // 2, (h - side) // 2
    image = image.crop((left, upper, left + side, upper + side))
    image = image.resize((768, 768))

    num_devices = jax.device_count()
    prompts = [prompt] * num_devices
    images = [image] * num_devices

    prompt_ids, processed_images = pipeline.prepare_inputs(prompts, images)
    _, _, height, width = processed_images.shape

    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)
    processed_images = shard(processed_images)

    prng_seed = jax.random.PRNGKey(0)
    prng_seed = jax.random.split(prng_seed, num_devices)

    output = pipeline(
        prompt_ids=prompt_ids,
        image=processed_images,
        params=p_params,
        prng_seed=prng_seed,
        strength=strength,
        num_inference_steps=num_inference_steps,
        height=int(height),
        width=int(width),
        jit=True,
    ).images
    output = output.reshape(-1, *output.shape[-3:])
    return pipeline.numpy_to_pil(np.array(output[0:1]))[0]


def shard(xs):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), xs
    )


if __name__ == "__main__":
    import argparse

    root = os.path.dirname(os.path.abspath(__file__))
    img_dir, out_base = os.path.join(root, "images"), os.path.join(root, "outputs_0.5")
    p = argparse.ArgumentParser()
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    paths = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    )


    prompts = ["gjgjfkjagkekgg", "apple", "ahhiodjfiogjdfogi", "Studio Ghibli style illustration"]
    pipe, params = load_pipeline()

    print(pipe.scheduler.config)

    for prompt in prompts:
        os.makedirs(os.path.join(out_base, prompt), exist_ok=True)
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            strength = args.strength
            out_path = os.path.join(out_base, prompt, f"{stem}_{strength:.2f}.png")
            out = img2img(pipe, params, path, prompt, strength, args.steps)
            out.save(out_path)
            print(out_path)
