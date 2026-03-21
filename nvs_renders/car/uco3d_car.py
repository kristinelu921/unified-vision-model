import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# to prepare the TFDS dataset
class UCO3DCar(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.1.0')
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
                'mask': tfds.features.Image(shape=(None, None, 3)),
                'image_latents': tfds.features.Tensor(shape=(128, 16, 16), dtype=tf.float32),
                'mask_latents': tfds.features.Tensor(shape=(128, 16, 16), dtype=tf.float32),
                'metadata': tfds.features.FeaturesDict({
                    'filename': tfds.features.Text(),
                    'original_resolution': tfds.features.Tensor(shape=(2,), dtype=tf.int32),
                    'image_path': tfds.features.Text(),
                    'plucker_path': tfds.features.Text(),
                    'extrinsics_w2c': tfds.features.Tensor(shape=(4, 4), dtype=tf.float32),
                    'intrinsics': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
                    'camera_json_path': tfds.features.Text(),
                })
            })
        )

    def _split_generators(self, dl_manager):
        path = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/nvs_renders/car'
        return {
            'train': self._generate_examples(img_dir=path+'/renders_car/train', plucker_dir=path+'/plucker_car/train',  latents_dir=path+'/car_train_latents'),
            'val': self._generate_examples(img_dir=path+'/renders_car/val', plucker_dir=path+'/plucker_car/val', latents_dir=path+'/car_val_latents'),
        }

    def _generate_examples(self, img_dir, plucker_dir, latents_dir):
        if not os.path.exists(img_dir):
            assert False, f"Image directory {img_dir} does not exist"
        if not os.path.exists(plucker_dir):
            assert False, f"Plucker directory {plucker_dir} does not exist"
        
        for filename in sorted(os.listdir(img_dir)):
            if filename.lower().endswith('.png') and not filename.endswith('_plucker.png'):
                img_path = os.path.join(img_dir, filename)
                image_latents_path = os.path.join(latents_dir, filename+'_image.npy')
                plucker_latents_path = os.path.join(latents_dir, filename+'_mask.npy')
                plucker_path = os.path.join(plucker_dir, filename.replace('.png', '_plucker.png'))
                car_dir = os.path.dirname(os.path.dirname(img_dir))
                camera_json_path = os.path.join(car_dir, 'camera_params.json')
                with open(camera_json_path, 'r') as f:
                    camera_data = json.load(f)
                views = camera_data['views']
                extrinsics_w2c = None
                intrinsics = None
                for view in views:
                    if view['image_file'] == filename:
                        extrinsics_w2c = view['extrinsics_world_to_camera']
                        intrinsics = view['intrinsics_ndc']
                        break
                if extrinsics_w2c is None or intrinsics is None:
                    continue
                try:
                    if not os.path.exists(plucker_path):
                        assert False, f"Plucker file {plucker_path} does not exist"
                    yield filename, {
                        'image': img_path,
                        'mask': plucker_path,
                        'image_latents': np.load(image_latents_path),
                        'mask_latents': np.load(plucker_latents_path),
                        'metadata': {
                            'filename': filename,
                            'original_resolution': list(Image.open(img_path).size),
                            'image_path': os.path.basename(img_path),
                            'plucker_path': os.path.basename(plucker_path),
                            'extrinsics_w2c': np.array(extrinsics_w2c, dtype=np.float32),
                            'intrinsics': np.array(intrinsics, dtype=np.float32),
                            'camera_json_path': camera_json_path,
                        },
                    }
                except:
                    with open('missing_plucker.txt', 'a') as f:
                        f.write(plucker_path + '\n')
                    continue

# data loader
def load_data(dataset_name='UCO3DCar', split='train', data_dir=None, batch_size=64, repeat=True, shuffle=True):
    load_kw = dict(split=split)
    if data_dir is not None:
        load_kw['data_dir'] = data_dir
    ds = tfds.load(dataset_name, **load_kw)
    ds = ds.map(lambda ex: (ex['image'], ex['mask'], ex['image_latents'], ex['mask_latents'], ex['metadata']))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# center crop + resize to 256 x 256
def preprocess_func(image, mask, image_latents, mask_latents, metadata): 
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    cropped_mask = tf.image.resize_with_crop_or_pad(mask, small_dim, small_dim)
    image = tf.image.resize(cropped_img, [256, 256], method='bilinear')
    mask = tf.image.resize(cropped_mask, [256, 256], method='nearest')
    return image, mask, image_latents, mask_latents, metadata

# sanity check visualization of dataloader
def sanity_check(ds_dir, split, num_batches=1, shuffle=False):
    dataloader = load_data(data_dir=ds_dir, split=split, batch_size=64, repeat=False, shuffle=shuffle)

    for i, (images, masks, image_latents, mask_latents, metadata) in enumerate(dataloader.take(num_batches)):
        print("reached sanity check")
        print("images.shape:", images.shape)
        print("masks.shape:", masks.shape)
        print("image_latents.shape:", image_latents.shape)
        print("mask_latents.shape:", mask_latents.shape)
        print("metadata:", metadata)
        print(f"--- batch {i+1} ---")
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")

        masks_15 = masks[:15].numpy() if hasattr(masks, 'numpy') else np.asarray(masks[:15])
        masks_15 = np.clip(masks_15, 0, 255).astype(np.uint8)

        

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
    ds_dir = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/uco3d_car_tfds/'
    sanity_check(ds_dir, 'train', num_batches=5)

def single_pass_thru():
    ds_dir = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/uco3d_car_tfds/'
    ds = load_data(data_dir=ds_dir, split='train', batch_size=64, repeat=False, shuffle=False)
    sanity_check(ds_dir, 'train', num_batches=5)
    with open('single_pass_thru.txt', 'w') as f:
        for image, mask, image_latents, mask_latents, metadata in ds:
            f.write(f"{image.shape}, {mask.shape}, {image_latents.shape}, {mask_latents.shape}, {metadata}\n")

if __name__ == "__main__":
    single_pass_thru()
