"""
Matrix Operations Utility
Provides reusable mathematical building blocks for custom PCA and SVD scripts.
"""
import numpy as np

def compute_eigen_decomposition(symmetric_matrix):
    """
    Computes eigenvalues and eigenvectors for a symmetric matrix (like a covariance matrix).
    Ensures they are sorted in descending order of variance.
    """
    try:
        # We use np.linalg.eigh because it is mathematically optimized 
        # and stable for symmetric/Hermitian matrices
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("🚨 Matrix decomposition failed to converge. Check for NaN/Inf values.")

    # Sort in descending order (highest variance first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]


def project_into_latent_space(X, components, num_dimensions):
    """
    Projects the normalized data matrix onto the top 'num_dimensions' principal components.
    """
    if num_dimensions > components.shape[1]:
        raise ValueError(f"Cannot select {num_dimensions} components from a matrix with {components.shape[1]} columns.")
        
    # Select the top k components (columns)
    selected_components = components[:, :num_dimensions]
    
    # Perform matrix multiplication to transform the data: (N x D) x (D x k) = (N x k)
    return np.dot(X, selected_components)