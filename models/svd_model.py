"""
Singular Value Decomposition (SVD) From Scratch
Uses reusable functions from matrix_ops utility with optimized computational paths.
"""
import numpy as np
from utils.matrix_ops import compute_eigen_decomposition

class SVDFromScratch:
    """
    Singular Value Decomposition (SVD) implemented from scratch.
    Decomposes a data matrix into U, Sigma, and V^T structural matrices.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.U = None
        self.singular_values = None
        self.VT = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Decomposes the feature matrix X into left/right singular vectors and singular values.
        """
        num_samples = X.shape[0]

        # 1. Compute V (Right Singular Vectors) using the Feature Gram Matrix: X^T * X (D x D)
        gram_features = np.dot(X.T, X)
        eigenvalues_V, eigenvectors_V = compute_eigen_decomposition(gram_features)
        
        # 2. Extract Singular Values: Square root of eigenvalues
        self.singular_values = np.sqrt(np.clip(eigenvalues_V, a_min=0, a_max=None))

        # 3. Compute U (Left Singular Vectors) stably and fast using: U = X * V / Sigma
        # This completely avoids calculating the massive (N x N) X * X^T matrix.
        # We handle zero singular values safely to avoid division by zero errors.
        nonzero_sigma = np.where(self.singular_values > 1e-8, self.singular_values, 1.0)
        U_full = np.dot(X, eigenvectors_V) / nonzero_sigma

        # 4. Truncate components to match target dimensional constraints
        self.U = U_full[:, :self.n_components]
        self.singular_values_truncated = self.singular_values[:self.n_components]
        self.VT = eigenvectors_V[:, :self.n_components].T

        # 5. Calculate Explained Variance 
        # For a standardized matrix, variance is proportional to (S^2) / (n - 1)
        variance_equiv = (self.singular_values ** 2) / (num_samples - 1)
        total_variance = np.sum(variance_equiv)
        
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(len(variance_equiv))
        else:
            self.explained_variance_ratio_ = variance_equiv / total_variance

        return self

    def transform(self, X):
        """
        Projects the data into the singular latent space using the right singular vectors (V).
        """
        if self.VT is None:
            raise ValueError("🚨 Model must be fitted before transforming data.")
        
        # Project using X * V (which is the transpose of V^T)
        return np.dot(X, self.VT.T)

    def fit_transform(self, X):
        """
        Fits the matrix elements and returns the truncated singular space coordinates.
        """
        self.fit(X)
        return self.transform(X)