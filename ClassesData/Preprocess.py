import os

import cv2
import numpy as np
import torch
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
    v2.RandomRotation(degrees=(-180,180),interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
]


def preprocess_sample(image_path, label_path,augment=False):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, :, :]

    label = cv2.resize(label, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (label > 127).astype(np.float32)
    mask = mask[np.newaxis, :, :]

    image_tensor = torch.from_numpy(image).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)

    n = len(transforms_pipeline)
    if augment:
        image_tensors = [image_tensor]
        mask_tensors = [mask_tensor]
        for transfo in transforms_pipeline:
            image_tensors.append(transfo(image_tensors[-1]))
            mask_tensors.append(transfo(mask_tensors[-1]))
        return image_tensors,mask_tensors
    return [image_tensor],[mask_tensor]

    



def build_tensors(split):

    pairs = collect_pairs(DATASET_ROOT, split, SENSORS)

    images = []
    masks = []

    for image_path, label_path in pairs:
        image_tensor, mask_tensor = preprocess_sample(image_path, label_path)
        
        #Data augmentation
        Augmented_img,Augmented_mask = preprocess_sample(image_path, label_path,augment=True)

        images.append(image_tensor[0])

        masks.append(mask_tensor[0])

        for i in range(len(Augmented_img)):
            images.append(Augmented_img[i])
            masks.append(Augmented_mask[i])

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