import numpy as np
import umap
from utils.data_loader import load_and_preprocess_data
from models.pca import CustomPCA
from models.svd import CustomSVD
from utils.visualizations import plot_side_by_side_variance, plot_latent_space_comparison

# 1. Load the dataset
X_scaled, y = load_and_preprocess_data("data/synthetic_data.csv", target_column="target_class")

# 2. Extract Custom PCA Matrix
pca = CustomPCA(n_components=10) # Fit all components for full scree comparison
X_pca = pca.fit_transform(X_scaled)

# 3. Extract Custom SVD Matrix
svd = CustomSVD(n_components=10)
X_svd = svd.fit_transform(X_scaled)

# 4. Generate Advanced Non-Linear UMAP projection for comparison
print("Computing non-linear manifold representation via UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 5. Output Graphics
plot_side_by_side_variance(pca.explained_variance_ratio_, svd.explained_variance_ratio_)
plot_latent_space_comparison(X_pca[:, :2], X_svd[:, :2], X_umap, y)

print("\n🎉 Model visualizations rendered successfully! Check your data/ folder.")