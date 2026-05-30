from utils.data_loader import load_and_preprocess_data
from models.pca import CustomPCA
import numpy as np

# 1. Load our synthetic dataset
X_scaled, y = load_and_preprocess_data("data/synthetic_data.csv", target_column="target_class")

# 2. Initialize our from-scratch PCA model to target 2 principal components
pca = CustomPCA(n_components=2)

# 3. Fit and transform the feature space
X_reduced = pca.fit_transform(X_scaled)

print("\n🚀 --- PCA From-Scratch Execution Test ---")
print(f"Original Data Matrix Shape: {X_scaled.shape}")
print(f"Reduced Latent Space Shape: {X_reduced.shape}")
print(f"Explained Variance Ratio for PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Explained Variance Ratio for PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Cumulative Variance Captured: {np.sum(pca.explained_variance_ratio_[:2])*100:.2f}%")
print("==========================================\n")