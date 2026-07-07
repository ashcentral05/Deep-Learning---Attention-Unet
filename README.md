# Deep Learning for Oil Spill Detection using Attention U-Net

This repository contains a PyTorch-based deep learning pipeline designed to detect oil spills from satellite Synthetic Aperture Radar (SAR) imagery (ALOS PALSAR and Sentinel-1A) using variants of the U-Net architecture.

---

## Repository Structure

<pre><code>Deep-Learning---Attention-Unet-mai
│  Data.py
│  environment.yml
│  evaluate.py
│  students_information.txt
│  train.py
│
├─ClassesData
│  │  DatasetLoader.py
│  │  Download.py
│  │  PairCheck.py
│  │  Preprocess.py
│  │
│  └─__pycache__
│          DatasetLoader.cpython-312.pyc
│          Download.cpython-312.pyc
│          PairCheck.cpython-312.pyc
│          Preprocess.cpython-312.pyc
│
├─ClassesML
│  │  Blocks.py
│  │  EarlyStopper.py
│  │  Scope.py
│  │  TrainerUNET.py
│  │  UNET.py
│  │  UNET_NoAttention.py
│  └─ __init__.py
│
├─Dataset
│  └─UNET
│          train_data_batches.pt
│          train_label_batches.pt
│          val_data_batches.pt
│          val_label_batches.pt
│
├─Model
│  └─20260707_143753
│          unet.pth
│
├─Result
│  └─YYMMDD_HHMMSS
│          loss.png
│          note.txt
│          visualize.png
│  
└─Utilities
    │  Utilities.py
    └─  __init__.py

---

### Getting Started

## 1. Installation & Environment Setup
This project uses Conda to manage package dependencies and ensure reproducibility. To recreate the exact environment, run the following commands in your terminal:

<pre><code># Create the environment from the configuration file
conda env create -f environment.yml </code></pre>

# Activate the environment
<pre><code>
conda activate oil_spill_env</code></pre>

## 2. Data Downloading, Preprocessing & Serialization

Before training, the dataset must be downloaded from Kaggle, and raw satellite image-mask pairs must be normalized, resized, and saved as serialized batches (`.pt` files).

To execute the entire data pipeline at once, run the `Data.py` script:
<pre><code>
python Data.py </code></pre>


Note: The pipeline resizes SAR images using bilinear interpolation and masks via nearest-neighbor interpolation. For augmented right-angle rotations, masks are explicitly re-thresholded to ensure boundaries stay strictly binary.   
There's a training set of 6455 images and a validation set of 1615 images. 5% of the training samples are augmented with random rotations, horisontal flip, and vertical flip，as a result, the final training set contains approximately 115% of the original data.  
This process may take **3-4 minutes**.  

### Advanced Options & Arguments
If the original dataset is too heavy for your local machine's RAM or GPU memory, or if you want to configure data augmentation, you can pass the following optional arguments through the terminal:

* --size &lt;int&gt;: Defines the spatial resolution size for image resizing (default is 256). If your computer has performance bottlenecks, you can generate smaller images to speed up processing and training.
* --augment: Enables data augmentation (rotations and flips) for the training set. By default, data augmentation is turned off (False).

#### Examples:
python Data.py --size 128
python Data.py --augment
python Data.py --size 128 --augment

## 3. Running the Training Loop
To train the main Attention U-Net model using the default configurations, execute the top-level script:

<pre><code>python train.py</code></pre>  
The trained model will be saved in Model/timestamp. The image of loss curves and validation curves will be saved in Result/timestamp.
---

## 4. Evaluation
To evaluate the trained models, execute the top-level script:

<pre><code>python evaluate.py YYMMDD_HHMMSS</code></pre>
**Please type the timestamp folder's name of the trained model you want to evaluate.**  
It will visualize the predictions, and save the image of visualization to Result/timestamp.
---

###  Key Architectural Features

- **Memory-Bound Pipeline:** Pre-computed batches are mapped directly into RAM via DatasetLoader.py at startup, heavily speeding up epoch execution times.
- **Geometric Invariance:** Robust data augmentation (synchronous flips and rotations) implemented within the processing flow to limit overfitting on limited radar patterns.
- **Multi-Scale Support:** The data loader adaptively switches between 128 x 128 and 256 x 256 spatial tensor files depending on available GPU memory constraints.
