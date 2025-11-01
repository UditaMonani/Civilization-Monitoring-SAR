# Civilization Monitoring using SAR and Optical Satellite Imagery

<p align="center">
  <img src="assets/7_final_comparison.png" alt="Final Comparison Plot" width="800"/>
</p>

This project presents a complete data science workflow for monitoring urbanization by fusing Sentinel-1 (SAR) and Sentinel-2 (Optical) satellite data. It uses a Random Forest classifier to process and classify imagery, automatically distinguishing between 'Urban' and 'Non-Urban' land cover.

---

## 🔹 Table of Contents
- [Project Overview](#-project-overview)
- [Technology Stack](#-technology-stack)
- [Key Features](#-key-features)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Results and Discussion](#-results-and-discussion)
- [Limitations and Future Scope](#-limitations-and-future-scope)

---

## 🔹 Project Overview

The core objective is to automate the detection of human settlements from satellite imagery. This is achieved by leveraging the unique strengths of two distinct data sources:
- **Sentinel-1 (SAR):** Synthetic Aperture Radar data excels at identifying geometric structures like buildings and infrastructure. Its all-weather, day-and-night capabilities make it a highly reliable data source.
- **Sentinel-2 (Optical):** High-resolution RGB imagery provides spectral information, which is used to approximate key environmental indices that help differentiate land cover types.

The script follows a standard machine learning pipeline: data loading, preprocessing (normalization, speckle filtering), feature extraction (approximated NDBI, NDVI), model training, and results visualization.

---

## 🔹 Technology Stack
- **Language:** Python 3
- **Core Libraries:** NumPy, Pandas, Scikit-learn
- **Image Processing:** Pillow, SciPy
- **Visualization:** Matplotlib, Seaborn

---

## 🔹 Key Features

- **Data Handling:** Processes separate, non-matching `.png` files from different source folders.
- **Image Standardization:** Automatically resizes optical images to match the dimensions of SAR images, ensuring pixel-wise feature alignment.
- **SAR Preprocessing:** Implements a Lee filter to reduce inherent speckle noise in Sentinel-1 data.
- **Feature Engineering:** Approximates NDBI (Normalized Difference Built-up Index) and NDVI (Normalized Difference Vegetation Index) from RGB channels to create powerful features for the classifier.
- **Machine Learning:** Implements and evaluates a Random Forest model for robust pixel-level classification.
- **Comprehensive Outputs:** Generates and saves multiple visualizations, including correlation matrices, confusion matrices, feature importance plots, and a final classification map.

---

## 🔹 Installation and Setup

Follow these instructions to set up and run the project on your local machine.

### Step 1: Clone the Repository
Clone this project to your local machine.
```bash
git clone https://github.com/YOUR_USERNAME/Civilization-Monitoring-SAR.git
cd Civilization-Monitoring-SAR
```

### Step 2: Install Dependencies
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
The data for this project is hosted on Kaggle and is **not** included in this repository.
- **Dataset Link:** [**Sentinel-1 & 2 Image Pairs (SAR & Optical)**](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/data)
- Download the dataset and unzip it on your computer.

### Step 4: Place the Data
Place the downloaded images into the correct project directories.
1.  Navigate to the unzipped dataset folder.
2.  Copy the `.png` images from the `Sentinel1/urban/` folder into this project's `data/urban/s1/` directory.
3.  Copy the `.png` images from the `Sentinel2/urban/` folder into this project's `data/urban/s2/` directory.

---

## 🔹 How to Run

Once the setup is complete, execute the main analysis script from the root directory of the project:

```bash
python run_analysis.py
```
The script will run the full pipeline, print the results to the console, and save all generated plots and tables to the `project_outputs/` directory.

---

## 🔹 Project Structure

The repository is organized as follows. The contents of `data/` and `project_outputs/` are ignored by Git.

```
├── .gitignore          # Specifies files for Git to ignore
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
├── run_analysis.py     # Main Python script for the analysis
│
├── assets/             # Contains images for the README
│   ├── 7_final_comparison.png
│   └── 6_feature_importance.png
│
├── data/
│   └── urban/
│       ├── s1/         # Folder for Sentinel-1 images (add data here)
│       └── s2/         # Folder for Sentinel-2 images (add data here)
│
└── project_outputs/    # All generated outputs will be saved here
```

---

## 🔹 Results and Discussion

The model successfully distinguishes between urban and non-urban areas, demonstrating the effectiveness of fusing SAR and optical data. The final classification map provides a clear visualization of civilization footprints in the analyzed region.

The feature importance plot below highlights that the **SAR backscatter intensity** (`S1_Intensity`) and the **approximated NDBI** were the most influential features in the model's decision-making process. This confirms our initial hypothesis that radar's sensitivity to structure and the built-up index are key differentiators.

<p align="center">
  <img src="assets/6_feature_importance.png" alt="Feature Importance Plot" width="600"/>
</p>

---

## 🔹 Limitations and Future Scope

- **Use GeoTIFF Data:** The primary limitation is the use of `.png` files, which lack the essential near-infrared (NIR) and short-wave infrared (SWIR) bands. The next step is to adapt this pipeline to use original GeoTIFF satellite files for scientifically accurate calculation of spectral indices.
- **Deep Learning Models:** A U-Net or other CNN-based segmentation model could be implemented for potentially higher accuracy and better spatial context awareness.
- **Time-Series Analysis:** The project could be extended to process images from multiple years to perform change detection and quantify the rate of urbanization over time.
- **Expand Dataset:** Future work could involve classifying more land-use types (e.g., water, barren land, forest) by using the full, multi-class dataset.