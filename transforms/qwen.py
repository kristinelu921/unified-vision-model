import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch_xla.core.xla_model as xm

device = xm.xla_device()
# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit", dtype=torch.bfloat16)
pipe = pipe.to(device)
xm.mark_step()

prompt = "Turn this cat into a dog"
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(image=input_image, prompt=prompt).images[0]