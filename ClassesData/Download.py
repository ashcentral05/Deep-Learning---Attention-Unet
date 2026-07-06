import os
import gdown
import kagglehub
import shutil


def download_dataset(path, from_drive=False, size=128):

    if from_drive:
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

    else:
        kagglehub.dataset_download("bitsandlayers/sar-oil-spill-segmentation-dataset-sos", output_dir=path)

path = os.path.join(os.getcwd(), "Dataset", "UNET")
downloaded_path = os.path.join(path, "dataset")
if downloaded_path and os.path.exists(downloaded_path):
    for item in os.listdir(downloaded_path):
        source_item = os.path.join(downloaded_path, item)
        target_item = os.path.join(path, item)
        
        shutil.move(source_item, target_item)
    
    dossier_auteur = os.path.join(path, "dataset")
    if os.path.exists(dossier_auteur):
        shutil.rmtree(dossier_auteur)