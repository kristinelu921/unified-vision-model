from uco3d_motor import load_data, sanity_check

def one_pass():
    ds_dir = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/uco3d_motor_tfds' #OR: if using a DIFFERENT ZONE (not us-central2), use '/mnt/klum/data'  (after you run the warmup script), see commands.txt for example

    sanity_check(ds_dir, 'train', num_batches = 1) # first sanity check to ensure the dataset is loading correctly

    ds = load_data(data_dir=ds_dir, split='train', batch_size=64, repeat=False, shuffle=False) #outputs the dataset in batches of 64 image-mask pairs (64, 256, 256, 3)

    with open('/kmh-nfs-ssd-us-mount/code/kristine/lvm/single_pass_thru.txt', 'w') as f:
        pass
    for i, (images, masks, image_latents, mask_latents, metadata) in enumerate(ds):
        # will loop through the entire dataset once, writing out the shape of each batch of image-mask pairs. images is a (64, 256, 256, 3) tensor, masks is a (64, 256, 256, 3) tensor.
        
        with open('/kmh-nfs-ssd-us-mount/code/kristine/lvm/single_pass_thru.txt', 'a') as f: #current command: write out the files
            f.write(f"{i}, {images.shape}, {masks.shape}, {image_latents.shape}, {mask_latents.shape}, {metadata}\n")
        ### FILL IN COMMANDS HERE ###

if __name__ == '__main__':
    one_pass()