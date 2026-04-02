from transformers import AutoModelForCausalLM


model_id = "/mnt/extracache/kristine/hunyuan"
# Currently we can not load the model using HF model_id `tencent/HunyuanImage-3.0-Instruct` directly
# due to the dot in the name.

kwargs = dict(
    attn_implementation="sdpa", 
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",   # Use "flashinfer" if FlashInfer is installed
    moe_drop_tokens=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

# Image-to-Image generation (TI2I)
prompt = "rant94jfksl"

input_img1 = "/kmh-nfs-ssd-us-mount/code/kristine/lvm/transforms/PB.jpeg"
imgs_input = [input_img1]

cot_text, samples = model.generate_image(
    prompt=prompt,
    image=imgs_input,
    seed=42,
    image_size="auto",
    use_system_prompt="en_unified",
    bot_task="None",  # Use "think_recaption" for reasoning and enhancement
    infer_align_image_size=True,  # Align output image size to input image size
    diff_infer_steps=50, 
    verbose=2
)

# Save the generated image
samples[0].save("image_edit.png")
