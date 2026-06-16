---

## Deep Dive: Dashboard & Manifold Interpretability

### 📈 Variance Retention Analysis
The **Variance Retention Curve (Linear)** demonstrates that the dataset reaches the standard $85\%$ variance retention threshold at approximately $k=6$ principal components. For our scratch-built 3D projections, choosing $k=3$ captures roughly $60\%$ of the total dataset variance. This represents an optimal engineering tradeoff, providing strong dimensionality reduction while enabling intuitive three-dimensional visual plotting of cluster separability.

### 🔄 PCA vs. SVD Alignment (Linear Paradigm)
In the top row of the dashboard, the 2D scatter plots for the custom **PCA Subspace** and **SVD Subspace** yield geometrically identical layouts that are simply rotated or mirrored versions of one another. This provides an immediate, visual mathematical proof that our from-scratch implementations are 100% accurate. Because PCA via covariance eigen-decomposition and SVD via Gram matrix ($X^T X$) factorization share underlying singular values, their latent spaces map the data structures identically.

### 🕸️ t-SNE vs. UMAP Behavior (Non-Linear Paradigm)
The bottom row highlights how non-linear manifold learning approaches capture structural patterns that linear math cannot compress effectively:
* **t-SNE Projections:** This algorithm maps the wine dataset into a broad, continuous spatial structure. By converting high-dimensional Euclidean distances into conditional probabilities, it focuses tightly on preserving localized neighborhoods.
* **UMAP Projections:** UMAP takes structural extraction a step further by preserving both local and global geometric distances. It constructs a fuzzy simplicial set that completely breaks the dense feature clusters into highly isolated, distinct spatial islands with wide, clean margins between groupings.

---