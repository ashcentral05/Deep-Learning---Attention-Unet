import os
from ClassesData.Download import *
from ClassesData.Preprocess import preprocess
import argparse


def main_data(size, augment):
    path_parent_project = os.getcwd()
    dataset_image_path = os.path.join(path_parent_project, "Dataset", "UNET")
    download_dataset(path=dataset_image_path)
    preprocess(size, augment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess SAR images for Attention U-Net."
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation for the training set (default: False)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Spatial resolution size for image resizing (default: 256)",
    )

    args = parser.parse_args()

    main_data(size=args.size, augment=args.augment)
