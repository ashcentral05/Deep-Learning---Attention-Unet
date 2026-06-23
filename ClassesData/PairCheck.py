import os
import glob


def find_image_label_pairs(sensor_dir):

    image_dir = os.path.join(sensor_dir, "image")
    label_dir = os.path.join(sensor_dir, "label")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))

    pairs = []
    missing = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        label_path = os.path.join(label_dir, filename)

        if os.path.exists(label_path):
            pairs.append((image_path, label_path))
        else:
            missing.append(image_path)

    if missing:
        raise ValueError(f"missing labels for: {missing}")

    return pairs


def collect_pairs(dataset_root, split, sensors):

    all_pairs = []
    for sensor in sensors:
        sensor_dir = os.path.join(dataset_root, split, sensor)
        all_pairs.extend(find_image_label_pairs(sensor_dir))

    return all_pairs