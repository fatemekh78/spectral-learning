import unittest
import numpy as np
from models.pca_model import PCAFromScratch
from models.svd_model import SVDFromScratch

class TestDecompositionModels(unittest.TestCase):
    def setUp(self):
        """Creates a standardized mock dataset for model testing."""
        np.random.seed(42)
        # 100 samples, 10 features
        self.X_mock = np.random.rand(100, 10) 
        # Standardize it manually for the tests
        self.X_mock = (self.X_mock - np.mean(self.X_mock, axis=0)) / np.std(self.X_mock, axis=0)
        self.n_components = 3

    def test_pca_from_scratch(self):
        """Verifies PCA initialization, dimension reduction, and variance calculations."""
        pca = PCAFromScratch(n_components=self.n_components)
        
        # Test fitting and transformation shapes
        X_reduced = pca.fit_transform(self.X_mock)
        
        # Output should be (100 samples, 3 components)
        self.assertEqual(X_reduced.shape, (100, self.n_components))
        
        # Components matrix should be (10 features, 3 components)
        self.assertEqual(pca.components_.shape, (10, self.n_components))
        
        # Explained variance ratio must be calculated and should sum to <= 1.0
        self.assertIsNotNone(pca.explained_variance_ratio_)
        self.assertTrue(np.sum(pca.explained_variance_ratio_) <= 1.0)

    def test_svd_from_scratch(self):
        """Verifies SVD initialization, dimension reduction, and structural matrices."""
        svd = svd = SVDFromScratch(n_components=self.n_components)
        
        # Test fitting and transformation shapes
        X_reduced = svd.fit_transform(self.X_mock)
        
        # Output should be (100 samples, 3 components)
        self.assertEqual(X_reduced.shape, (100, self.n_components))
        
        # Check structural matrices U, Sigma, and V^T
        self.assertEqual(svd.U.shape, (100, self.n_components))
        self.assertEqual(len(svd.singular_values_truncated), self.n_components)
        self.assertEqual(svd.VT.shape, (self.n_components, 10))
        
        # Explained variance ratio must be calculated and should sum to <= 1.0
        self.assertIsNotNone(svd.explained_variance_ratio_)
        self.assertTrue(np.sum(svd.explained_variance_ratio_) <= 1.0)

if __name__ == '__main__':
    unittest.main()