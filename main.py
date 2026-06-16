import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Internal absolute imports based on  project structure
from utils.data_loader import load_wine_quality_data
from models.pca_model import PCAFromScratch as CustomPCA
from models.svd_model import SVDFromScratch as CustomSVD
from utils.clustering import apply_kmeans
from models.advanced_models import compute_nonlinear_embeddings

def main():
    print("====================================================================")
    print("   INITIALIZING SPECTRAL LEARNING & DIMENSIONALITY REDUCTION PLATFORM")
    print("====================================================================\n")

    # Step 1: Data Resiliency & Processing Pipeline
    # Loader automatically handles missing local files and handles live web fallbacks
    X, y, feature_names = load_wine_quality_data()  
    # Standarization to ensure zero-mean and unit variance across dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[Step 1] Data preprocessing complete. Matrix dimensions: {X_scaled.shape}")

    # Step 2: Custom Linear Spectral Learning (From Scratch)
    k_dimensions = 3
    print(f"\n[Step 2] Executing custom algorithms from scratch (Target Latent Subspace: k={k_dimensions})...")
    
    # Custom PCA Execution via Covariance Eigen-Decomposition
    pca_model = CustomPCA(n_components=k_dimensions)
    X_pca = pca_model.fit_transform(X_scaled)
    pca_variance = pca_model.explained_variance_ratio_
    print(f" -> Custom PCA Complete. Captured Variance (Top {k_dimensions}): {sum(pca_variance)*100:.2f}%")

    # Custom SVD Execution via Gram Matrix (X^T * X) Factorization
    svd_model = CustomSVD(n_components=k_dimensions)
    X_svd = svd_model.fit_transform(X_scaled)
    svd_variance = svd_model.explained_variance_ratio_
    print(f" -> Custom SVD Complete. Captured Variance (Top {k_dimensions}): {sum(svd_variance)*100:.2f}%")

    # Step 3: Run Advanced Non-Linear Reductions (t-SNE & UMAP)
    # Explores complex data manifolds for resume-level benchmarking
    X_tsne, X_umap = compute_nonlinear_embeddings(X_scaled)

  # Step 4: Downstream Latent Space Clustering & Validation
    print("\n[Step 4] Fitting K-Means Partitioning across Dimensional Subspaces...")
    n_clusters = 3
    
    # Extract ['labels'] from the dictionary returned by apply_kmeans
    print("Evaluating Custom PCA...")
    pca_results = apply_kmeans(X_pca, n_clusters=n_clusters, random_state=42)
    pca_labels = pca_results['labels']

    print("Evaluating Custom SVD...")
    svd_results = apply_kmeans(X_svd, n_clusters=n_clusters, random_state=42)
    svd_labels = svd_results['labels']

    # Process non-linear embeddings to showcase manifold clustering robustness
    print("Evaluating Advanced t-SNE...")
    tsne_results = apply_kmeans(X_tsne, n_clusters=n_clusters, random_state=42)
    tsne_labels = tsne_results['labels']

    print("Evaluating Advanced UMAP...")
    umap_results = apply_kmeans(X_umap, n_clusters=n_clusters, random_state=42)
    umap_labels = umap_results['labels']

    # Step 5: Master UI Visualization Dashboard Generation (Expanded 2x3 Layout)
    print("\n[Step 5] Launching Comprehensive Graphical Analysis Interface...")
    
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle("Linear vs. Non-Linear Dimensionality Reduction Benchmark Dashboard", 
                 fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Variance Retention Curves (PCA vs SVD Alignment)
    ax1 = fig.add_subplot(2, 3, 1)
    full_pca_model = CustomPCA(n_components=X_scaled.shape[1])
    full_pca_model.fit(X_scaled)
    cum_variance = np.cumsum(full_pca_model.explained_variance_ratio_)
    
    ax1.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='-', color='b', label='PCA Cumulative')
    ax1.plot(range(1, len(cum_variance) + 1), cum_variance, marker='s', linestyle='--', color='r', alpha=0.5, label='SVD Cumulative')
    ax1.axhline(y=0.85, color='orange', linestyle=':', label='85% Retention Cutoff')
    ax1.set_title("Variance Retention Curves (Linear)", fontweight='bold')
    ax1.set_xlabel("Number of Selected Components")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')

    # Plot 2: Custom 2D PCA Mapping
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=pca_labels, cmap='viridis', s=25, alpha=0.7, edgecolors='none')
    ax2.set_title("K-Means on Custom PCA Subspace (PC1 vs PC2)", fontweight='bold')
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    fig.colorbar(sc2, ax=ax2, label="Cluster Labels")

    # Plot 3: Custom 2D SVD Mapping
    ax3 = fig.add_subplot(2, 3, 3)
    sc3 = ax3.scatter(X_svd[:, 0], X_svd[:, 1], c=svd_labels, cmap='plasma', s=25, alpha=0.7, edgecolors='none')
    ax3.set_title("K-Means on Custom SVD Subspace (SVD1 vs SVD2)", fontweight='bold')
    ax3.set_xlabel("SVD Component 1")
    ax3.set_ylabel("SVD Component 2")
    fig.colorbar(sc3, ax=ax3, label="Cluster Labels")

    # Plot 4: Custom 3D Projection Layout
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    sc4 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=pca_labels, cmap='viridis', s=20, alpha=0.7)
    ax4.set_title("3D Latent Space Projection (Top 3 PCs)", fontweight='bold')
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.set_zlabel("PC 3")

    # Plot 5: Advanced Non-Linear t-SNE Embeddings (Preserves Local Structural Manifolds)
    ax5 = fig.add_subplot(2, 3, 5)
    sc5 = ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tsne_labels, cmap='coolwarm', s=25, alpha=0.7, edgecolors='none')
    ax5.set_title("Advanced t-SNE Projections", fontweight='bold')
    ax5.set_xlabel("t-SNE Dimension 1")
    ax5.set_ylabel("t-SNE Dimension 2")
    fig.colorbar(sc5, ax=ax5, label="Cluster Labels")

    # Plot 6: Advanced Non-Linear UMAP Embeddings (Preserves Global/Local Balance)
    ax6 = fig.add_subplot(2, 3, 6)
    sc6 = ax6.scatter(X_umap[:, 0], X_umap[:, 1], c=umap_labels, cmap='inferno', s=25, alpha=0.7, edgecolors='none')
    ax6.set_title("Advanced UMAP Projections", fontweight='bold')
    ax6.set_xlabel("UMAP Dimension 1")
    ax6.set_ylabel("UMAP Dimension 2")
    fig.colorbar(sc6, ax=ax6, label="Cluster Labels")

    plt.tight_layout()
    print("\n[Success] Dashboard initialized safely. Close graph window to cleanly exit pipeline.")
    plt.show()

if __name__ == "__main__":
    main()