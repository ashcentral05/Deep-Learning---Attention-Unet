import os
import cv2
import numpy as np
import torch
import shutil
from torchvision.transforms import v2
from PairCheck import collect_pairs

IMAGE_SIZE = (256, 256)
SENSORS = ["palsar", "sentinel"]
BATCH_SIZE = 16

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "Dataset", "UNET")
OUTPUT_DIR = DATASET_ROOT


transforms_pipeline = [
    v2.RandomChoice([v2.RandomRotation(degrees=(180, 180), interpolation=v2.InterpolationMode.BILINEAR),
                     v2.RandomRotation(degrees=(90, 90), interpolation=v2.InterpolationMode.BILINEAR),
                     v2.RandomRotation(degrees=(-180, -180), interpolation=v2.InterpolationMode.BILINEAR),
                     v2.RandomRotation(degrees=(-90, -90), interpolation=v2.InterpolationMode.BILINEAR)]
                     )
    ,
    v2.RandomHorizontalFlip(p=1.0),
    v2.RandomVerticalFlip(p=1.0),
]


def preprocess_sample(image_path, label_path, augment=False):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, :, :]

    label = cv2.resize(label, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (label > 127).astype(np.float32)
    mask = mask[np.newaxis, :, :]

    image_tensor = torch.from_numpy(image)
    mask_tensor = torch.from_numpy(mask)

    n = len(transforms_pipeline)
    if augment:
        image_tensors = []
        mask_tensors = []
        for transfo in transforms_pipeline:
            img_transfo, mask_transfo = transfo(
                image_tensor, mask_tensor
            )  # we apply the same transformation on both the mask and the image
            mask_transfo = (mask_transfo > 0.5).to(
                torch.float32
            )  # due to the interpolation, we need the apply a threshold on the transformed mask
            image_tensors.append(img_transfo)
            mask_tensors.append(mask_transfo)
        return image_tensors, mask_tensors
    return [image_tensor], [mask_tensor]


def build_tensors(split, is_train):

    pairs = collect_pairs(DATASET_ROOT, split, SENSORS)

    images = []
    masks = []

    for idx, (image_path, label_path) in enumerate(pairs):
        image_tensor, mask_tensor = preprocess_sample(image_path, label_path)

        images.append(image_tensor[0])
        masks.append(mask_tensor[0])

        if (
            is_train and idx % 4 == 0
        ):  # Data augmentation is made on 25% of the data in average.
            Augmented_img, Augmented_mask = preprocess_sample(
                image_path, label_path, augment=True
            )

            for i in range(len(Augmented_img)):
                images.append(Augmented_img[i])
                masks.append(Augmented_mask[i])

    return torch.stack(images, dim=0), torch.stack(masks, dim=0)


def make_batches(tensor, batch_size):
    return [tensor[i : i + batch_size] for i in range(0, tensor.shape[0], batch_size)]


def preprocess():
    train_images, train_masks = build_tensors("train", is_train=True)
    val_images, val_masks = build_tensors("test", is_train=False)

    train_data_batches = make_batches(train_images, BATCH_SIZE)
    train_label_batches = make_batches(train_masks, BATCH_SIZE)
    val_data_batches = make_batches(val_images, BATCH_SIZE)
    val_label_batches = make_batches(val_masks, BATCH_SIZE)

    torch.save(train_data_batches, os.path.join(OUTPUT_DIR, "train_data_batches.pt"))
    torch.save(train_label_batches, os.path.join(OUTPUT_DIR, "train_label_batches.pt"))
    torch.save(val_data_batches, os.path.join(OUTPUT_DIR, "val_data_batches.pt"))
    torch.save(val_label_batches, os.path.join(OUTPUT_DIR, "val_label_batches.pt"))

    folders_to_delete = [".complete", "test", "train"]

    for folder in folders_to_delete:
        full_path = os.path.join(OUTPUT_DIR, folder)

        # Check if the folder exists before attempting deletion
        if os.path.exists(full_path) and os.path.isdir(full_path):
            try:
                shutil.rmtree(full_path)
            except Exception as e:
                print(f"Error while deleting {full_path}: {e}")
        else:
            print(f"Folder does not exist (already deleted): {full_path}")
