"""
Clustering Utility Module
Applies K-Means clustering algorithm configurations and extracts validation metrics.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def apply_kmeans(X_reduced: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> dict:
    """
    Applies K-Means clustering on low-dimensional data spaces and returns core validation metrics.
    
    Parameters:
    -----------
    X_reduced : numpy.ndarray
        The matrix array representing data projected onto lower latent components.
    n_clusters : int, default=3
        The specified number of discrete cluster divisions to establish.
    random_state : int, default=42
        Determines random number generation for centroid initialization reproducibility.
        
    Returns:
    --------
    metrics : dict
        A compiled dictionary containing predicted cluster labels, model inertia,
        and mathematical separability metrics.
    """
    try:
        # Confirm array contains correct dimensionality profiles
        if X_reduced.ndim != 2:
            raise ValueError(f"Expected 2D matrix structure, received array with shape {X_reduced.shape}")
            
        # Instantiate and fit the standard K-Means clustering engine
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_reduced)
        
        # Calculate separation scores safely (only if more than 1 distinct cluster layer is generated)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            sil = silhouette_score(X_reduced, labels)
            db = davies_bouldin_score(X_reduced, labels)
        else:
            sil, db = 0.0, float('inf')
            print("[WARNING] Clustering resulted in a single distinct class label matrix layer.")

        return {
            "labels": labels,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
            "inertia": kmeans.inertia_
        }
        
    except Exception as error:
        raise RuntimeError(f"Critical execution error encountered during clustering processing: {str(error)}")