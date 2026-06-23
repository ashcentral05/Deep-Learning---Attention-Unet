import os

import cv2
import numpy as np
import torch

from PairCheck import collect_pairs

IMAGE_SIZE = (256, 256)
SENSORS = ["palsar", "sentinel"]
BATCH_SIZE = 16

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "Dataset", "UNET")
OUTPUT_DIR = DATASET_ROOT


def preprocess_sample(image_path, label_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, :, :]

    label = cv2.resize(label, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (label > 127).astype(np.float32)
    mask = mask[np.newaxis, :, :]

    return torch.from_numpy(image), torch.from_numpy(mask)


def build_tensors(split):

    pairs = collect_pairs(DATASET_ROOT, split, SENSORS)

    images = []
    masks = []

    for image_path, label_path in pairs:
        image_tensor, mask_tensor = preprocess_sample(image_path, label_path)
        images.append(image_tensor)
        masks.append(mask_tensor)

    return torch.stack(images, dim=0), torch.stack(masks, dim=0)


def make_batches(tensor, batch_size):
    return [tensor[i:i + batch_size] for i in range(0, tensor.shape[0], batch_size)]


def main():

    train_images, train_masks = build_tensors("train")
    val_images, val_masks = build_tensors("test")

    train_data_batches = make_batches(train_images, BATCH_SIZE)
    train_label_batches = make_batches(train_masks, BATCH_SIZE)
    val_data_batches = make_batches(val_images, BATCH_SIZE)
    val_label_batches = make_batches(val_masks, BATCH_SIZE)

    torch.save(train_data_batches, os.path.join(OUTPUT_DIR, "train_data_batches.pt"))
    torch.save(train_label_batches, os.path.join(OUTPUT_DIR, "train_label_batches.pt"))
    torch.save(val_data_batches, os.path.join(OUTPUT_DIR, "val_data_batches.pt"))
    torch.save(val_label_batches, os.path.join(OUTPUT_DIR, "val_label_batches.pt"))


if __name__ == "__main__":
    main()