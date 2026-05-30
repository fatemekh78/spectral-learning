import numpy as np
# Import your reusable mathematical backend
from utils.matrix_ops import compute_eigen_decomposition, project_into_latent_space

class CustomPCA:
    """
    Principal Component Analysis (PCA) implemented completely from scratch.
    Extracts orthogonal axes maximizing global data variance.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        Computes the covariance matrix and finds the optimal principal component axes.
        """
        num_samples = X.shape[0]
        
        # 1. Calculate Covariance Matrix from scratch: Sigma = (X^T * X) / (N - 1)
        # Note: X must already be mean-centered and scaled via our data loader!
        covariance_matrix = np.dot(X.T, X) / (num_samples - 1)
        
        # 2. Leverage our reusable math engine for eigen-decomposition
        self.eigenvalues, self.components = compute_eigen_decomposition(covariance_matrix)
        
        # 3. Calculate Explained Variance Ratio (Rubric Requirement)
        # Ratio = Individual Eigenvalue / Sum of all Eigenvalues
        total_variance = np.sum(self.eigenvalues)
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(len(self.eigenvalues))
        else:
            self.explained_variance_ratio_ = self.eigenvalues / total_variance
            
        return self

    def transform(self, X):
        """
        Projects the scaled input matrix into the reduced-dimensional latent space.
        """
        if self.components is None:
            raise ValueError("🚨 Model must be fitted before transforming data.")
            
        # Leverage our reusable projection logic
        return project_into_latent_space(X, self.components, self.n_components)

    def fit_transform(self, X):
        """
        Convenience wrapper to fit and transform in a single pass.
        """
        return self.fit(X).transform(X)