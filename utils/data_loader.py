import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath, target_column=None):
    """
    Loads dataset, handles missing values, and strictly normalizes the feature matrix.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"🚨 Error: Could not locate {filepath}. Check your data folder.")

    # 1. Handle Null Values (Drop rows with missing data for mathematical stability)
    df = df.dropna()

    # 2. Separate Features (X) from Target/Labels (y)
    if target_column and target_column in df.columns:
        y = df[target_column].values
        X_raw = df.drop(columns=[target_column]).values
    else:
        y = None
        X_raw = df.values

    # 3. Mean Centering & Scaling (From Scratch)
    # PCA requires data to have a mean of 0 and standard deviation of 1
    mean_vec = np.mean(X_raw, axis=0)
    std_vec = np.std(X_raw, axis=0)
    
    # Prevent division by zero if a feature has no variance
    std_vec[std_vec == 0] = 1e-8 
    
    X_scaled = (X_raw - mean_vec) / std_vec

    print(f"✅ Data processed successfully. Matrix shape: {X_scaled.shape}")
    return X_scaled, y