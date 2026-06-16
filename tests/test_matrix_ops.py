import unittest
import numpy as np
from utils.matrix_ops import compute_eigen_decomposition, project_into_latent_space

class TestMatrixOps(unittest.TestCase):
    def test_eigen_decomposition_sorting(self):
        """Ensures eigenvalues are sorted in descending order."""
        # Create a simple symmetric matrix
        symmetric_matrix = np.array([[2, 1], [1, 2]])
        
        eigenvalues, eigenvectors = compute_eigen_decomposition(symmetric_matrix)
        
        # Check if the first eigenvalue is greater than or equal to the second
        self.assertTrue(eigenvalues[0] >= eigenvalues[1])
        # Check shapes
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (2, 2))

    def test_project_into_latent_space(self):
        """Ensures projection reduces dimensions correctly."""
        X_mock = np.random.rand(10, 5) # 10 samples, 5 features
        components_mock = np.random.rand(5, 5) # 5 components
        
        # Project down to 2 dimensions
        X_projected = project_into_latent_space(X_mock, components_mock, num_dimensions=2)
        
        self.assertEqual(X_projected.shape, (10, 2))

    def test_projection_error_handling(self):
        """Ensures projection fails safely if asked for too many dimensions."""
        X_mock = np.random.rand(10, 5)
        components_mock = np.random.rand(5, 3) # Only 3 components exist
        
        with self.assertRaises(ValueError):
            project_into_latent_space(X_mock, components_mock, num_dimensions=4)

if __name__ == '__main__':
    unittest.main()