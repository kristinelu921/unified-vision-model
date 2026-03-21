from PIL import Image
import os

palette = [
    [0,0,0], #black, object
    [255,0,0], #red, human
    [64,34,64], #purplish, building
    [0,255,0], #green, vegetation
    [93,64,0], #brown, ground
    [0,0,255], #blue, sky
    [0,255,255], #cyan, water
]

# convert annotation image (Image) to segmentation mask
def process_annotation(anno_image):
    new_anno_image = Image.new('RGB', anno_image.size)
    pixels = anno_image.load()
    new_pixels = new_anno_image.load()
    for y in range(anno_image.size[1]):
        for x in range(anno_image.size[0]):
            pixel_value = pixels[x, y]
            new_pixels[x, y] = tuple(palette[pixel_value])
    return new_anno_image

# visualize annotation image (Image) and original image (Image) side by side
def visualize_annotations(anno_path, image_root, output_path):
    anno_image = Image.open(anno_path)
    new_anno_image = process_annotation(anno_image)

    anno_path_tail = anno_path.split('/')[-1]
    image_left = Image.open(os.path.join(image_root, anno_path_tail.replace('.png', '.jpg')))
    image_right = new_anno_image

    side_by_side = Image.new('RGB', (image_left.width + image_right.width, image_left.height))
    side_by_side.paste(image_left, (0, 0))
    side_by_side.paste(image_right, (image_left.width, 0))
    side_by_side.save(output_path)
    return side_by_side


if __name__ == '__main__':
    anno_root = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/mass13k_data/masks/annotations/val'
    image_root = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/mass13k_data/val/val'
    output_root = '/kmh-nfs-ssd-us-mount/code/kristine/lvm/mass13k_data/visualized/val_annotations'
    os.makedirs(output_root, exist_ok=True)

    for file in os.listdir(anno_root)[:10]:
        if file.endswith('.png'):
            anno_path = os.path.join(anno_root, file)
            output_path = os.path.join(output_root, file)
            visualize_annotations(anno_path, image_root, output_path)
