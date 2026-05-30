import numpy as np
from utils.data_loader import load_and_preprocess_data
from utils.matrix_ops import compute_eigen_decomposition

# 1. Load and normalize the high-dimensional data we generated
X_scaled, y = load_and_preprocess_data("data/synthetic_data.csv", target_column="target_class")

# 2. Compute a manual covariance matrix to test our math engine
# Covariance Formula: (X^T * X) / (N - 1)
covariance_matrix = np.dot(X_scaled.T, X_scaled) / (X_scaled.shape[0] - 1)

# 3. Pass it to our reusable math function
eigenvalues, eigenvectors = compute_eigen_decomposition(covariance_matrix)

print(f"📊 Test Results:")
print(f"Covariance Matrix Shape: {covariance_matrix.shape}")
print(f"Top 3 Sorted Eigenvalues: {eigenvalues[:3]}")
print("Everything is connected perfectly!")