
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from visualize_annotations import process_annotation

# to prepare the TFDS dataset
class Mass13k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image_latent': tfds.features.Tensor(shape=(128, 16, 16), dtype=tf.float32),
                'mask_latent': tfds.features.Tensor(shape=(128, 16, 16), dtype=tf.float32), #pixel mask
                'metadata': tfds.features.FeaturesDict({
                    'filename': tfds.features.Text(),
                    'original_resolution': tfds.features.Tensor(shape=(2,), dtype=tf.int32), # [width, height]
                    'image_path': tfds.features.Text(),
                    'mask_path': tfds.features.Text(),
                })
            })
        )

    def _split_generators(self, dl_manager):
        path = '/kmh-nfs-ssd-us-mount/code/xtiange/mass13k_latents' #TODO: replace with GSBUCKET location (us: central2)
        return {
            'train': self._generate_examples(img_latent_dir=path + '/train/images/', mask_latent_dir=path+'/train/masks/'),
            'val': self._generate_examples(img_latent_dir=path+'/val/images/', mask_latent_dir=path+'/val/masks/'),
            'test': self._generate_examples(img_latent_dir=path+'/test/images/', mask_latent_dir=path+'/test/masks/'),
        }

    def _generate_examples(self, img_latent_dir, mask_latent_dir):
        print(img_latent_dir)
        print(mask_latent_dir)
        if not os.path.exists(img_latent_dir):
            assert False, f"Image directory {img_latent_dir} does not exist"
        if not os.path.exists(mask_latent_dir):
            assert False, f"Mask directory {mask_latent_dir} does not exist"
        
        for filename in sorted(os.listdir(img_latent_dir)):
            #print("REACHED, filename:", filename)
            if filename.lower().endswith('.npy'):
                img_latent_path = os.path.join(img_latent_dir, filename)
                mask_latent_path = os.path.join(mask_latent_dir, filename)
                split = os.path.basename(os.path.dirname(img_latent_dir.rstrip('/')))
                res_path = os.path.join(
                    '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_data/images',
                    split,
                    filename.rsplit('.', 1)[0],   # keeps .jpg from *.jpg.npy
                )
                print("REACHED, res_path:", res_path)
                try:
                    if not os.path.exists(mask_latent_path):
                        assert False, f"Mask file {mask_latent_path} does not exist"
                    yield filename, {
                        'image_latent': np.load(img_latent_path),
                        'mask_latent': np.load(mask_latent_path),
                        'metadata': {
                            'filename': filename,
                            'original_resolution': list(Image.open(res_path).size),  # [width, height] from PIL
                            'image_path': os.path.basename(img_latent_path).rsplit('.', 1)[0],
                            'mask_path': os.path.basename(mask_latent_path).rsplit('.', 1)[0],
                        }, #metadata is a dictionary of the filename, original resolution, image path, and mask path
                    }
                except:
                    with open('missing_masks.txt', 'a') as f:
                        f.write(mask_latent_path + '\n')
                    continue

# data loader
# Returns batched dataset: each batch is (images, masks) with shapes (B, 256, 256, 3)
def load_data(dataset_name='Mass13k', split='train', data_dir=None, batch_size=64, repeat=True, shuffle=True):
    load_kw = dict(split=split)
    if data_dir is not None:
        load_kw['data_dir'] = data_dir
    ds = tfds.load(dataset_name, **load_kw)
    ds = ds.map(lambda ex: (ex['image_latent'], ex['mask_latent'], ex['metadata']))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# set steps
# this was for original image mask pairs
#center crop + resize to 256 x 256
# def preprocess_func(image, mask, metadata): 
#     shape = tf.shape(image)
#     small_dim = tf.reduce_min([shape[0], shape[1]])
#     cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
#     cropped_mask = tf.image.resize_with_crop_or_pad(mask, small_dim, small_dim)
#     image = tf.image.resize(cropped_img, [256, 256], method='bilinear')
#     mask = tf.image.resize(cropped_mask, [256, 256], method='nearest')
#     return image, mask, metadata

# sanity check visualization of dataloader
def sanity_check(ds_dir, split, num_batches=1, shuffle=False):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"sanity_check_{date_str}_{split}.txt"

    dataloader = load_data(data_dir=ds_dir, split=split, batch_size=64, repeat=False, shuffle=shuffle)

    with open(out_path, 'w') as f:
        f.write(f"Sanity check run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (split={split})\n\n")
        for i, (image_latents, mask_latents, metadata) in enumerate(dataloader.take(num_batches)):
            img_lat = image_latents.numpy() if hasattr(image_latents, 'numpy') else np.asarray(image_latents)
            msk_lat = mask_latents.numpy() if hasattr(mask_latents, 'numpy') else np.asarray(mask_latents)

            f.write(f"--- batch {i+1} ---\n")
            f.write(f"Image latent shape: {img_lat.shape}\n")
            f.write(f"Image latent mean: {np.mean(img_lat):.6f}, std: {np.std(img_lat):.6f}\n")
            f.write(f"Mask latent shape: {msk_lat.shape}\n")
            f.write(f"Mask latent mean: {np.mean(msk_lat):.6f}, std: {np.std(msk_lat):.6f}\n")
            f.write(f"metadata: {metadata}\n")

    print(f"Saved: {out_path}")

def main():
    ds_dir = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_latents_tfds/' # use the root directory
    sanity_check(ds_dir, 'train', num_batches=5)

def single_pass_thru():
    ds_dir = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_latents_tfds/' # use the root directory
    ds = load_data(data_dir=ds_dir, split='train', batch_size=64, repeat=False, shuffle=False)
    with open('single_pass_thru.txt', 'w') as f:
        for image_latent, mask_latent, metadata in ds:
            f.write(f"{image_latent.shape}, {mask_latent.shape}, {metadata}\n")

if __name__ == "__main__":
    ds_dir = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_latents_tfds/'
    sanity_check(ds_dir, 'train', num_batches=5)
    single_pass_thru()