# -*- coding: utf-8 -*-
"""
Civilization Monitoring using SAR and Optical Data - Local Version (v2)

This script is updated to handle separate, non-matching filenames in the S1 and S2 directories.
"""

# =============================================================================
# 1. SETUP AND CONFIGURATION
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- Civilization Monitoring Project Initialized (V2) ---")

# --- IMPORTANT: UPDATE THESE PATHS ---
# Base path where the 'urban' folder is located
BASE_DATA_PATH = r"E:\CAIR\SAR_Optical_Dataset\v_2\urban"

# Define paths for Sentinel-1 (SAR) and Sentinel-2 (Optical) images
S1_FOLDER_PATH = os.path.join(BASE_DATA_PATH, "s1")
S2_FOLDER_PATH = os.path.join(BASE_DATA_PATH, "s2")

# Define path for saving outputs
OUTPUT_PATH = "project_outputs"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"Created output directory at: {OUTPUT_PATH}")

# =============================================================================
# 2. DATA LOADING AND STANDARDIZATION
# =============================================================================
print("\n--- Step 2: Loading Data and Standardizing Dimensions ---")

try:
    # Find the first PNG file in the S1 directory
    s1_files = [f for f in os.listdir(S1_FOLDER_PATH) if f.lower().endswith('.png')]
    if not s1_files:
        raise FileNotFoundError("No PNG files found in the S1 directory.")
    s1_filename = s1_files[0]
    s1_image_path = os.path.join(S1_FOLDER_PATH, s1_filename)
    
    # Find the first PNG file in the S2 directory
    s2_files = [f for f in os.listdir(S2_FOLDER_PATH) if f.lower().endswith('.png')]
    if not s2_files:
        raise FileNotFoundError("No PNG files found in the S2 directory.")
    s2_filename = s2_files[0]
    s2_image_path = os.path.join(S2_FOLDER_PATH, s2_filename)

    print(f"Loading S1 image: {s1_filename}")
    print(f"Loading S2 image: {s2_filename}")

    # Load images using Pillow
    s1_pil = Image.open(s1_image_path).convert('L') # Load S1 as grayscale
    s2_pil = Image.open(s2_image_path).convert('RGB') # Load S2 as RGB

    # --- CRITICAL STEP: RESIZE S2 TO MATCH S1 DIMENSIONS ---
    print(f"Original S1 dimensions: {s1_pil.size}")
    print(f"Original S2 dimensions: {s2_pil.size}")
    
    if s1_pil.size != s2_pil.size:
        print(f"Resizing S2 image to match S1 dimensions ({s1_pil.size})...")
        s2_pil_resized = s2_pil.resize(s1_pil.size, Image.Resampling.LANCZOS)
    else:
        s2_pil_resized = s2_pil
    
    # Convert PIL images to NumPy arrays for processing
    s1_image = np.array(s1_pil)
    s2_image_rgb = np.array(s2_pil_resized)
    
    print(f"Final S1 image shape for processing: {s1_image.shape}")
    print(f"Final S2 image shape for processing: {s2_image_rgb.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure your folder paths are correct and contain at least one PNG file in each directory.")
    exit()

# --- Visualize Sample Images ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(s1_image, cmap='gray')
axes[0].set_title('Sentinel-1 SAR Image')
axes[0].axis('off')
axes[1].imshow(s2_image_rgb)
axes[1].set_title('Sentinel-2 Optical Image (Resized)')
axes[1].axis('off')
plt.suptitle('Initial Loaded Data')
plt.savefig(os.path.join(OUTPUT_PATH, '1_loaded_images.png'))
plt.show()

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n--- Step 3: Preprocessing Data ---")

def normalize(image):
    """Normalizes an image to a 0-1 scale."""
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val > 0:
        return (image - min_val) / (max_val - min_val)
    return image

def lee_filter(img, size):
    """Applies a Lee filter to reduce speckle noise."""
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-8)
    return img_mean + img_weights * (img - img_mean)

# Normalize images
s1_norm = normalize(s1_image)
s2_norm = normalize(s2_image_rgb)

# Apply speckle filter to the normalized SAR image
s1_filtered = lee_filter(s1_norm, size=5)

# --- Visualize Preprocessing ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(s1_norm, cmap='gray')
axes[0].set_title('Normalized S1 SAR Image')
axes[0].axis('off')
axes[1].imshow(s1_filtered, cmap='gray')
axes[1].set_title('Speckle-Filtered S1 SAR Image')
axes[1].axis('off')
plt.suptitle('SAR Image Preprocessing')
plt.savefig(os.path.join(OUTPUT_PATH, '2_preprocessing_effect.png'))
plt.show()

# =============================================================================
# 4. FEATURE EXTRACTION
# =============================================================================
print("\n--- Step 4: Extracting Features ---")
print("WARNING: NDVI/NDBI calculation is an APPROXIMATION due to missing NIR/SWIR bands in PNG files.")

# Separate RGB channels from the normalized S2 image
red_ch = s2_norm[:, :, 0]
green_ch = s2_norm[:, :, 1]

# --- APPROXIMATE Spectral Indices ---
nir_approx = red_ch
swir_approx = green_ch
epsilon = 1e-8

ndvi = (nir_approx - red_ch) / (nir_approx + red_ch + epsilon)
ndbi = (swir_approx - nir_approx) / (swir_approx + nir_approx + epsilon)
ndwi = (green_ch - nir_approx) / (green_ch + nir_approx + epsilon)

# --- Visualize Indices ---
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
# Visualization code remains the same...
im1 = axes[0].imshow(ndvi, cmap='RdYlGn')
axes[0].set_title('Approximated NDVI (Vegetation)')
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1)
im2 = axes[1].imshow(ndbi, cmap='viridis')
axes[1].set_title('Approximated NDBI (Built-up)')
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1)
im3 = axes[2].imshow(ndwi, cmap='Blues')
axes[2].set_title('Approximated NDWI (Water)')
axes[2].axis('off')
fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.1)
plt.suptitle('Approximated Spectral Indices')
plt.savefig(os.path.join(OUTPUT_PATH, '3_spectral_indices.png'))
plt.show()

# --- Prepare Feature Matrix ---
features_stacked = np.stack([
    s1_filtered.flatten(),
    ndvi.flatten(),
    ndbi.flatten(),
    ndwi.flatten()
], axis=1)

feature_names = ['S1_Intensity', 'NDVI', 'NDBI', 'NDWI']
df = pd.DataFrame(features_stacked, columns=feature_names)

print("Feature DataFrame head:")
print(df.head())

# --- Plot Correlation Matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig(os.path.join(OUTPUT_PATH, '4_feature_correlation.png'))
plt.show()

# =============================================================================
# 5. MODEL DEVELOPMENT (RANDOM FOREST)
# =============================================================================
print("\n--- Step 5: Developing Classification Model ---")

# Generate "pseudo-labels" based on our approximated NDBI
ndbi_threshold = 0.05
labels = (ndbi.flatten() > ndbi_threshold).astype(int)
df['label'] = labels

print(f"Generated {np.sum(labels == 1)} Urban pixels and {np.sum(labels == 0)} Non-Urban pixels.")

# --- Train and Evaluate Model ---
X = df[feature_names]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("Training complete.")

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Urban', 'Urban']))

# --- Plot Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Urban', 'Urban'], yticklabels=['Non-Urban', 'Urban'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(OUTPUT_PATH, '5_confusion_matrix.png'))
plt.show()

# =============================================================================
# 6. VISUALIZATION AND RESULTS
# =============================================================================
print("\n--- Step 6: Visualizing Results ---")

# --- Plot Feature Importance ---
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.savefig(os.path.join(OUTPUT_PATH, '6_feature_importance.png'))
plt.show()

# --- Generate Full Classification Map ---
print("Generating full classification map...")
full_prediction = rf_model.predict(df[feature_names])
classification_map = full_prediction.reshape(s1_image.shape)
print("Map generated.")

# --- Visualize Final Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(s2_image_rgb)
axes[0].set_title('Original Sentinel-2 RGB Image ("Before")')
axes[0].axis('off')
im = axes[1].imshow(classification_map, cmap='YlOrRd')
axes[1].set_title('Urban Area Classification Map ("After")')
axes[1].axis('off')
cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1], orientation='horizontal', pad=0.1)
cbar.ax.set_xticklabels(['Non-Urban', 'Urban'])
plt.suptitle('Civilization Monitoring Results')
plt.savefig(os.path.join(OUTPUT_PATH, '7_final_comparison.png'))
plt.show()

# =============================================================================
# 7. FINAL SUMMARY
# =============================================================================
print("\n--- Step 7: Final Summary and Conclusion ---")

total_pixels = classification_map.size
urban_pixels = np.sum(classification_map == 1)
urban_percentage = (urban_pixels / total_pixels) * 100

print(f"\nQuantitative Analysis:")
print(f"  - Total pixels in the image: {total_pixels}")
print(f"  - Pixels classified as Urban: {urban_pixels}")
print(f"  - Percentage of area classified as Urban: {urban_percentage:.2f}%")

print("\nConclusion:")
print("This project demonstrated a workflow for land-use classification using separate SAR and Optical PNGs.")
print("The script successfully handles different filenames and image sizes by resizing the optical image.")
print("For scientific applications, using original GeoTIFF files with all spectral bands is crucial.")

print("\n--- PROJECT COMPLETE ---")
print(f"All outputs have been saved to the '{OUTPUT_PATH}' directory.")