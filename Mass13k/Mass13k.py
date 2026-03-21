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
                'image': tfds.features.Image(shape=(None, None, 3)),
                'mask': tfds.features.Image(shape=(None, None, 3)), #pixel mask
                'metadata': tfds.features.FeaturesDict({
                    'filename': tfds.features.Text(),
                    'original_resolution': tfds.features.Tensor(shape=(2,), dtype=tf.int32), # [width, height]
                    'image_path': tfds.features.Text(),
                    'mask_path': tfds.features.Text(),
                })
            })
        )

    def _split_generators(self, dl_manager):
        path = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_data' #TODO: replace with GSBUCKET location (us: central2)
        image_path_train = path+'/images/train'
        print(image_path_train)
        return {
            'train': self._generate_examples(img_dir=image_path_train, mask_dir=path+'/annotations/train'),
            'val': self._generate_examples(img_dir=path+'/images/val', mask_dir=path+'/annotations/val'),
            'test': self._generate_examples(img_dir=path+'/images/test', mask_dir=path+'/annotations/test'),
        }

    def _generate_examples(self, img_dir, mask_dir):
        if not os.path.exists(img_dir):
            assert False, f"Image directory {img_dir} does not exist"
        if not os.path.exists(mask_dir):
            assert False, f"Mask directory {mask_dir} does not exist"
        
        for filename in sorted(os.listdir(img_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, filename)
                mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
                try:
                    if not os.path.exists(mask_path):
                        assert False, f"Mask file {mask_path} does not exist"
                    yield filename, {
                        'image': img_path,
                        'mask': mask_path,
                        'metadata': {
                            'filename': filename,
                            'original_resolution': list(Image.open(img_path).size),  # [width, height] from PIL
                            'image_path': os.path.basename(img_path),
                            'mask_path': os.path.basename(mask_path),
                        }, #metadata is a dictionary of the filename, original resolution, image path, and mask path
                    }
                except:
                    with open('missing_masks.txt', 'a') as f:
                        f.write(mask_path + '\n')
                    continue

# data loader
# Returns batched dataset: each batch is (images, masks) with shapes (B, 256, 256, 3)
def load_data(dataset_name='Mass13k', split='train', data_dir=None, batch_size=64, repeat=True, shuffle=True):
    load_kw = dict(split=split)
    if data_dir is not None:
        load_kw['data_dir'] = data_dir
    ds = tfds.load(dataset_name, **load_kw)
    ds = ds.map(lambda ex: (ex['image'], ex['mask'], ex['metadata']))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# set steps

#center crop + resize to 256 x 256
def preprocess_func(image, mask, metadata): 
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    cropped_mask = tf.image.resize_with_crop_or_pad(mask, small_dim, small_dim)
    image = tf.image.resize(cropped_img, [256, 256], method='bilinear')
    mask = tf.image.resize(cropped_mask, [256, 256], method='nearest')
    return image, mask, metadata

# sanity check visualization of dataloader
def sanity_check(ds_dir, split, num_batches=1, shuffle=False):

    dataloader = load_data(data_dir=ds_dir, split=split, batch_size=64, repeat=False, shuffle=shuffle)

    for i, (images, masks, metadata) in enumerate(dataloader.take(num_batches)):
        print("reached sanity check")
        print("images.shape:", images.shape)
        print("metadata:", metadata)
        m = masks[:15].numpy() if hasattr(masks, 'numpy') else np.asarray(masks[:15])
        masks_15 = np.stack([np.array(process_annotation(Image.fromarray(m[k, :, :, 0].astype(np.uint8), mode='L'))) for k in range(min(15, len(m)))])

        print(f"--- batch {i+1} ---")
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")

        print(f"Image max/min: {np.max(images)}, {np.min(images)}")
        print(f"Mask max/min: {np.max(masks)}, {np.min(masks)}")

        plt.figure(figsize=(10, 12))
        for j in range(3):
            plt.subplot(3, 2, 2*j+1)
            img = np.clip(np.asarray(images[j]), 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(3, 2, 2*j+2)
            plt.imshow(masks_15[j])
            plt.axis('off')
        
        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"sanity_check_{date_str}_{split}_batch_{i+1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()

def main():
    ds_dir = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_tfds/' # use the root directory
    sanity_check(ds_dir, 'train', num_batches=5)

def single_pass_thru():
    ds_dir = '/kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_tfds/' # use the root directory
    ds = load_data(data_dir=ds_dir, split='train', batch_size=64, repeat=False, shuffle=False)
    sanity_check(ds_dir, 'train', num_batches=5)
    with open('single_pass_thru.txt', 'w') as f:
        for image, mask, metadata in ds:
            f.write(f"{image.shape}, {mask.shape}, {metadata}\n")

if __name__ == "__main__":
    single_pass_thru()
    