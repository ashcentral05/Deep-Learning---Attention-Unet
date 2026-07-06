import os
from ClassesData.Download import *
from ClassesData.Preprocess import preprocess

def main_data():
    path_parent_project = os.getcwd()
    dataset_image_path = os.path.join(path_parent_project, "Dataset", "UNET")
    download_dataset(path=dataset_image_path)
    preprocess()

if __name__ == "__main__":
    main_data()

