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

## 2. Data Downloading, Preprocessing & Serialisation
Before training, the dataset must be downloaded from Kaggle and raw satellite pairs must be normalized, resized, and saved as serialized binaries (.pt files). To do all of this at the same time, run the file `Data.py`.

<pre><code>python Data.py</code></pre>

*Note: The script resizes images using bilinear interpolation and masks via nearest-neighbor interpolation to preserve strictly binary labels.*

## 3. Running the Training Loop
To train the main Attention U-Net model using the default configurations, execute the top-level script:

<pre><code>python UNET.py</code></pre>

---

###  Key Architectural Features

- **Memory-Bound Pipeline:** Pre-computed batches are mapped directly into RAM via DatasetLoader.py at startup, heavily speeding up epoch execution times.
- **Geometric Invariance:** Robust data augmentation (synchronous flips and rotations) implemented within the processing flow to limit overfitting on limited radar patterns.
- **Multi-Scale Support:** The data loader adaptively switches between 128 x 128 and 256 x 256 spatial tensor files depending on available GPU memory constraints.
