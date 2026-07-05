import os
import gdown


def download_dataset(path, size=128):
    """
    Downloads the dataset files from Google Drive if they do not already exist in the specified directory.
    """
    dataset_path = path
    files_to_download = {
        "train_data_batches128.pt": "1RCi5UeItYTlwhJsC88GwB8STcTVWUyDC",
        "val_data_batches128.pt": "1QJLd6jxaAivr7B8nr-E-mJVZ1nYQGoKK",
        "train_label_batches128.pt": "1yiGHi_mEfT3ZUUYm6O8SCRLEMWp_wjbp",
        "val_label_batches128.pt": "1ZspbELQuYvqz82YG7MdXuUNdzZ_sjH7O",
    }

    print("Dowloading dataset...")

    for file, id_drive in files_to_download.items():
        final_path = os.path.join(dataset_path, file)

        # We don't download the file if it already exists
        if not os.path.exists(final_path):
            print(f"Dowloading {file}...")
            url = f"https://drive.google.com/uc?id={id_drive}"
            gdown.download(url, final_path, quiet=False)
        else:
            print(f"{file} already exists, skipping download.")

    print("Download completed.")


download_dataset(path=os.path.join(os.getcwd(), "Dataset", "UNET"), size=128)
