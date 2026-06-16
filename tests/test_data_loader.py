import os
import unittest
import numpy as np
import pandas as pd
import tempfile
from utils.data_loader import load_wine_quality_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Creates a temporary CSV file with mock data before each test."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.mock_file_path = os.path.join(self.test_dir.name, "mock_wine.csv")
        
        # Create a tiny mock dataframe with a missing value (NaN) to test cleaning
        mock_data = {
            'fixed acidity': [7.4, 7.8, 7.8, np.nan, 7.4],
            'volatile acidity': [0.70, 0.88, 0.76, 0.28, 0.66],
            'quality': [5, 5, 5, 6, 5]
        }
        df = pd.DataFrame(mock_data)
        df.to_csv(self.mock_file_path, index=False)

    def tearDown(self):
        """Cleans up the temporary directory after tests."""
        self.test_dir.cleanup()

    def test_loading_and_cleaning(self):
        """Tests if data loads, handles NaNs, and splits features/targets correctly."""
        X_norm, y, features = load_wine_quality_data(file_path=self.mock_file_path)
        
        # Check shapes (5 samples, 2 features, 1 target)
        self.assertEqual(X_norm.shape, (5, 2))
        self.assertEqual(y.shape, (5,))
        self.assertEqual(len(features), 2)
        
        # Check if NaN was handled (there should be no NaNs in the output matrix)
        self.assertFalse(np.isnan(X_norm).any())

    def test_normalization_math(self):
        """Tests if the Z-score normalization correctly forces mean=0 and std=1."""
        X_norm, _, _ = load_wine_quality_data(file_path=self.mock_file_path)
        
        means = np.mean(X_norm, axis=0)
        stds = np.std(X_norm, axis=0)
        
        # Use np.allclose to handle tiny floating-point rounding differences
        self.assertTrue(np.allclose(means, 0.0, atol=1e-7), "Mean is not 0")
        self.assertTrue(np.allclose(stds, 1.0, atol=1e-7), "Standard deviation is not 1")

if __name__ == '__main__':
    unittest.main()