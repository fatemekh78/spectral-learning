"""
Data Loader Utility
Handles dataset acquisition, local file caching, cleaning, and normalization.
"""
import os
import pandas as pd
import numpy as np

def load_wine_quality_data(file_path: str = "data/wine_quality.csv") -> tuple:
    """
    Loads the Wine Quality dataset. Checks locally first, otherwise downloads from UCI.
    
    Returns:
    --------
    X_normalized : numpy.ndarray (Standardized features)
    y : numpy.ndarray (Target labels)
    feature_names : list (Column headers)
    """
    # 1. Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created directory: {directory}")

    # 2. Check if file exists, if not, download it
    if not os.path.exists(file_path):
        print(f"[INFO] {file_path} not found. Downloading from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        try:
            df = pd.read_csv(url, sep=';')
            df.to_csv(file_path, index=False)
            print(f"[INFO] Successfully downloaded and cached at: {file_path}")
        except Exception as e:
            raise IOError(f"[ERROR] Could not download dataset: {e}")
    else:
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded dataset from local file: {file_path}")

    # 3. Data Cleaning (Handling missing values)
    if df.isnull().values.any():
        print(f"[WARNING] Missing values detected. Imputing with mean.")
        df = df.fillna(df.mean())

    # 4. Preparation
    feature_cols = [c for c in df.columns if c != 'quality']
    X = df[feature_cols].to_numpy()
    y = df['quality'].to_numpy()
    
    # 5. Normalization (Standardization)
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    std[std == 0] = 1e-8  # Prevent division by zero
    X_normalized = (X - mean) / std
    
    return X_normalized, y, feature_cols