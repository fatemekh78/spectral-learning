"""
Data Loader Utility
Handles dataset acquisition, cleaning, missing values, and matrix normalization.
"""

import pandas as pd
import numpy as np

def load_wine_quality_data(url: str = None) -> tuple:
    """
    Loads the Wine Quality dataset, cleans it, and prepares matrices.
    
    Parameters:
    -----------
    url : str, optional
        URL or local filepath to the dataset csv. Defaults to UCI repository link.
        
    Returns:
    --------
    X_normalized : numpy.ndarray
        High-dimensional numerical feature matrix normalized to mean=0, std=1.
    y : numpy.ndarray
        Target labels (wine quality scores).
    feature_names : list
        Names of the chemical features.
    """
    if url is None:
        # Defaulting to the Red Wine Quality dataset from UCI Repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        # Step 1: Read data (the wine dataset uses ';' as a delimiter)
        df = pd.read_csv(url, sep=';')
        print(f"[INFO] Successfully loaded dataset. Shape: {df.shape}")
    except Exception as e:
        raise IOError(f"[ERROR] Failed to load dataset from {url}: {str(e)}")
        
    # Step 2: Data Cleaning (Handle missing values dynamically)
    missing_sum = df.isnull().sum().sum()
    if missing_sum > 0:
        print(f"[WARNING] Found {missing_sum} missing values. Imputing with column means.")
        df = df.fillna(df.mean())
    else:
        print("[INFO] Data cleaning check passed: No missing values detected.")

    # Step 3: Matrix Preparation
    # Isolate target variable ('quality') from features
    feature_columns = [col for col in df.columns if col != 'quality']
    feature_names = feature_columns.copy()
    
    X_raw = df[feature_columns].to_numpy()
    y = df['quality'].to_numpy()
    
    # Step 4: Standardization from Scratch (Z-score normalization)
    # Calculate mean and standard deviation along the columns (axis=0)
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    
    # Avoid potential division by zero if a feature has zero variance
    std[std == 0] = 1e-8
    
    # Apply standardizing mathematical operation
    X_normalized = (X_raw - mean) / std
    
    print("[INFO] Matrix preparation complete. Feature matrix standardized successfully.")
    return X_normalized, y, feature_names

if __name__ == "__main__":
    # Quick standalone testing for structural verification
    try:
        X, y, features = load_wine_quality_data()
        print(f"Verified Matrix Target Format: Matrix Shape={X.shape}, Labels Shape={y.shape}")
        print(f"Verified Feature Bounds: Calculated Mean={np.mean(X, axis=0)[0]:.4f} (Expected: 0)")
    except Exception as error:
        print(f"Testing failed: {error}")