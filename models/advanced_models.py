import numpy as np
from sklearn.manifold import TSNE
import umap

def compute_nonlinear_embeddings(X_scaled):
    """
    Computes t-SNE and UMAP embeddings for high-dimensional feature matrices.
    """
    print("\n[Step 4.5] Computing Advanced Non-Linear Embeddings...")
    
    # t-SNE Implementation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f" -> t-SNE complete. Latent shape: {X_tsne.shape}")
    
    # UMAP Implementation
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)
    print(f" -> UMAP complete. Latent shape: {X_umap.shape}")
    
    return X_tsne, X_umap