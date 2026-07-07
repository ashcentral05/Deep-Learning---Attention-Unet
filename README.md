# Deep Learning for Oil Spill Detection using Attention U-Net

This repository contains a PyTorch-based deep learning pipeline designed to detect oil spills from satellite Synthetic Aperture Radar (SAR) imagery (ALOS PALSAR and Sentinel-1A) using variants of the U-Net architecture.

---

## Repository Structure

<pre><code>├── ClassesData/
│   ├── DatasetLoader.py       # Parametric multi-scale tensor loader & data pipelines
│   ├── Download.py            # Utility script for dataset downloading
│   ├── PairCheck.py           # Integrity script checking image-mask consistency
│   └── Preprocess.py          # Serialisation, resizing (256x256), and normalization pipeline
│
├── ClassesML/
│   ├── EarlyStopper.py        # Early stopping logic to prevent overfitting
│   ├── Scope.py               # Performance and evaluation scoping functions
│   ├── TrainerClassifier.py   # Main training, validation, and optimization loop
│   ├── UNET_NoAttention.py    # Standard baseline U-Net implementation
│   ├── UNET_V1.py             # Optimized or custom version of the model
│   └── Visualization.py       # Metrics plotters and segmentation mask outputs
│
├── Dataset/UNET/              # Serialized tensor binaries (loaded directly into RAM)
│   ├── train_data_batches.pt / train_data_batches128.pt
│   ├── train_label_batches.pt / train_label_batches128.pt
│   ├── val_data_batches.pt / val_data_batches128.pt
│   └── val_label_batches.pt / val_label_batches128.pt
│
├── Models/                    # Directory reserved for saved model weights (.pth)
├── Utilities/                 # Helper scripts and miscellaneous tools
│
├── environment.yml            # Conda environment configuration file
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── README.md                  # Project documentation (this file)
├── UNET.py                    # Main executable script to launch experiments
├── students_information.txt   # Team credentials and project info
└── *_log.txt                  # Error and runtime execution logs</code></pre>

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

### Advanced Options & Arguments
If the original dataset is too heavy for your local machine's RAM or GPU memory, or if you want to configure data augmentation, you can pass the following optional arguments through the terminal:

* --size &lt;int&gt;: Defines the spatial resolution size for image resizing (default is 256). If your computer has performance bottlenecks, you can generate smaller images to speed up processing and training.
* --augment: Enables data augmentation (rotations and flips) for the training set. By default, data augmentation is turned off (False).

#### Examples:
python Data.py --size 128
python Data.py --augment
python Data.py --size 128 --augment

Note: The pipeline resizes SAR images using bilinear interpolation and masks via nearest-neighbor interpolation. For augmented right-angle rotations, masks are explicitly re-thresholded to ensure boundaries stay strictly binary.</code></pre>

## 3. Running the Training Loop
To train the main Attention U-Net model using the default configurations, execute the top-level script:

<pre><code>python train.py</code></pre>

---

## 4. Evaluation
To evaluate the trained models, execute the top-level script:

<pre><code>python evaluate.py</code></pre>
---

###  Key Architectural Features

- **Memory-Bound Pipeline:** Pre-computed batches are mapped directly into RAM via DatasetLoader.py at startup, heavily speeding up epoch execution times.
- **Geometric Invariance:** Robust data augmentation (synchronous flips and rotations) implemented within the processing flow to limit overfitting on limited radar patterns.
- **Multi-Scale Support:** The data loader adaptively switches between 128 x 128 and 256 x 256 spatial tensor files depending on available GPU memory constraints.
