"""
Principal Component Analysis (PCA) From Scratch
Uses reusable functions from matrix_ops utility.
"""
import numpy as np
from utils.matrix_ops import compute_eigen_decomposition, project_into_latent_space

class PCAFromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Calculates the principal components using your custom eigen decomposition.
        """
        # Step 1: Calculate the covariance matrix from scratch
        n_samples = X.shape[0]
        covariance_matrix = np.dot(X.T, X) / (n_samples - 1)
        
        # Step 2: Use your reusable function to get sorted eigenvalues/vectors
        eigenvalues, eigenvectors = compute_eigen_decomposition(covariance_matrix)
        
        # Step 3: Store ONLY the top k components and calculate variance profiles
        self.components_ = eigenvectors[:, :self.n_components]
        
        total_variance = np.sum(eigenvalues)
        variance_ratio = eigenvalues / (total_variance if total_variance > 0 else 1e-8)
        self.explained_variance_ratio_ = variance_ratio[:self.n_components]
        
        return self

    def transform(self, X):
        """
        Transforms the data using your reusable projection function.
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before transformation.")
            
        return project_into_latent_space(X, self.components_, self.n_components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)