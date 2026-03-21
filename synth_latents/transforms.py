import torch


def generate_channel_mask(before, after, mask):

    if mask is None:
        return after

    channel_count = before.shape[1]

    if isinstance(mask, torch.Tensor):
        channel_mask = mask.to(device=before.device)
        if channel_mask.dtype != torch.bool:
            channel_mask = channel_mask.bool()
    else:
        channel_mask = torch.zeros(channel_count, dtype=torch.bool, device=before.device)
        channel_mask[list(mask)] = True

    channel_mask = channel_mask.view(1, channel_count, *([1] * (before.ndim - 2)))
    return torch.where(channel_mask, after, before)


def add_noise_to_latents(latents, noise_level=0.1, seed=42, channel_mask=None):
    generator = torch.Generator(device=latents.device).manual_seed(seed)
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    noisy_latents = latents + noise_level * noise
    noisy_latents = generate_channel_mask(latents, noisy_latents, channel_mask)

    with open("latent_stats.txt", "a") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"noise_level: {noise_level}, max: {noise.max().item()}, min: {noise.min().item()}\n")
        f.write(f"latents max: {latents.max().item()}, min: {latents.min().item()}\n")
        f.write(f"noise max: {noise.max().item()}, min: {noise.min().item()}\n")
        f.write(
            f"latents + noise max: {noisy_latents.max().item()}, min: {noisy_latents.min().item()}\n"
        )
    return noisy_latents


def affine_transform_latents(latents, shift_level=0.1, scale_level=1.0):
    return latents * scale_level + shift_level

def interpolate_latents(latents_1, latents_2, alpha=0.5):
    return latents_1 * (1 - alpha) + latents_2 * alpha

def set_dimesion_to_zero(latents, dimension):
    #1, 4, 32, 32
    latents[dimension]=0
    return latents

def apply_function_to_latents(latents, function, *args, **kwargs):
    if function == "shift":
        return affine_transform_latents(latents, *args, **kwargs)
    if function == "scale":
        return affine_transform_latents(latents, *args, **kwargs)
    if function == "noise":
        return add_noise_to_latents(latents, *args, **kwargs)
    return latents
