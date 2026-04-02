# LVIS TFDS

The TFDS builder has four main parts:

## 1. Source JSON parsing

The builder reads the LVIS JSON and uses:

- `images`
- `annotations`
- `categories`

It groups annotations by `image_id`, so each image gets all object instances that belong to it.

## 2. Per-image example construction

Each TFDS example is one image plus its object annotations.

Each annotation contains:

- `bbox`
- `area`
- `category_id`
- `class_idx`
- `category_name`
- `iscrowd`

`category_id` is the original LVIS class id.  
`class_idx` is a contiguous class index derived from the sorted LVIS category ids.

Example annotation:

```python
{
  "bbox": [267.06, 254.69, 108.05, 291.9],   # shape (4,)
  "area": 18648.27,                          # shape ()
  "category_id": 1139,                       # shape ()
  "class_idx": 1138,                         # shape ()
  "category_name": "vase",                   # scalar string
  "iscrowd": 0,                              # shape ()
}
```

Per-image annotation shapes:

- `bbox`: `(num_objects, 4)`
- `area`: `(num_objects,)`
- `category_id`: `(num_objects,)`
- `class_idx`: `(num_objects,)`
- `category_name`: `(num_objects,)`
- `iscrowd`: `(num_objects,)`

So if one image has 8 objects, its `bbox` shape is `(8, 4)`.

Categories are also stored in each example as the full LVIS category table.

Each category entry contains:

- `category_id`
- `class_idx`
- `category_name`

Example category entry:

```python
{
  "category_id": 1139,      # shape ()
  "class_idx": 1138,        # shape ()
  "category_name": "vase",  # scalar string
}
```

Category table shapes:

- `category_id`: `(num_categories,)`
- `class_idx`: `(num_categories,)`
- `category_name`: `(num_categories,)`

## 3. Sampling

The current builder shuffles images with a fixed seed and yields every 3rd image.

So each split is a reproducible one-third sample of LVIS.

## 4. Loading and batching

At load time:

- images are center-cropped/resized to `256 x 256`
- annotations are batched with `ragged_batch(...)`

Ragged batching is needed because different images have different numbers of boxes.
