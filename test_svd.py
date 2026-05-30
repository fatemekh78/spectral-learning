from utils.data_loader import load_and_preprocess_data
from models.svd import CustomSVD
import numpy as np

# 1. Load data
X_scaled, y = load_and_preprocess_data("data/synthetic_data.csv", target_column="target_class")

# 2. Initialize from-scratch SVD
svd = CustomSVD(n_components=2)
X_reduced_svd = svd.fit_transform(X_scaled)

print("\n⚙️ --- SVD From-Scratch Execution Test ---")
print(f"Original Data Matrix Shape: {X_scaled.shape}")
print(f"Reduced SVD Latent Space Shape: {X_reduced_svd.shape}")
print(f"SVD Variance Ratio Component 1: {svd.explained_variance_ratio_[0]:.4f}")
print(f"SVD Variance Ratio Component 2: {svd.explained_variance_ratio_[1]:.4f}")
print(f"Cumulative SVD Variance Captured: {np.sum(svd.explained_variance_ratio_[:2])*100:.2f}%")
print("==========================================\n")