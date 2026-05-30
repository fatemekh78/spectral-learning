import numpy as np
from utils.matrix_ops import compute_eigen_decomposition

class CustomSVD:
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

        # 1. Compute V (Right Singular Vectors) using the Gram Matrix: X^T * X
        gram_features = np.dot(X.T, X)
        eigenvalues_V, eigenvectors_V = compute_eigen_decomposition(gram_features)
        
        # 2. Extract Singular Values: Square root of eigenvalues
        # We enforce a floor at 0 to dodge rounding math quirks causing negative values
        self.singular_values = np.sqrt(np.clip(eigenvalues_V, a_min=0, a_max=None))

        # 3. Compute U (Left Singular Vectors) using the Outer Product Matrix: X * X^T
        gram_samples = np.dot(X, X.T)
        _, eigenvectors_U = compute_eigen_decomposition(gram_samples)

        # 4. Truncate components to match target dimensional constraints
        self.U = eigenvectors_U[:, :self.n_components]
        self.singular_values_truncated = self.singular_values[:self.n_components]
        self.VT = eigenvectors_V[:, :self.n_components].T

        # 5. Calculate Explained Variance (Square of Singular Values represents variance)
        squared_singular = self.singular_values ** 2
        total_variance = np.sum(squared_singular)
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(len(squared_singular))
        else:
            self.explained_variance_ratio_ = squared_singular / total_variance

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
        return self.fit(X).transform(X)