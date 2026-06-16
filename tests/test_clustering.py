import unittest
import numpy as np
from utils.clustering import apply_kmeans

class TestClustering(unittest.TestCase):
    def setUp(self):
        """Creates a mock 2D reduced dataset for clustering."""
        np.random.seed(42)
        # Create two distinct blobs of data
        blob1 = np.random.normal(loc=0, scale=0.5, size=(20, 2))
        blob2 = np.random.normal(loc=5, scale=0.5, size=(20, 2))
        self.X_reduced = np.vstack([blob1, blob2])

    def test_apply_kmeans_success(self):
        """Verifies k-means returns the correct dictionary structure and types."""
        results = apply_kmeans(self.X_reduced, n_clusters=2)
        
        self.assertIn("labels", results)
        self.assertIn("silhouette_score", results)
        self.assertIn("davies_bouldin_score", results)
        
        # Check that it assigned exactly 40 labels (one for each sample)
        self.assertEqual(len(results["labels"]), 40)
        # Silhouette score should be valid since there are clear blobs
        self.assertTrue(0.0 <= results["silhouette_score"] <= 1.0)

    def test_apply_kmeans_dimension_error(self):
        """Ensures clustering throws an error if given a 1D array instead of 2D matrix."""
        bad_array = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(RuntimeError):
            apply_kmeans(bad_array, n_clusters=2)

if __name__ == '__main__':
    unittest.main()