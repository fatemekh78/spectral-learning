import pandas as pd
import numpy as np

def make_high_dimensional_data():
    print("Generating synthetic high-dimensional dataset...")
    np.random.seed(42)
    
    # Create 150 samples with 10 features, grouped into 3 distinct hidden classes
    num_samples = 150
    num_features = 10
    
    X = np.random.randn(num_samples, num_features)
    
    # Add artificial structural groupings (classes) so PCA/SVD have patterns to find
    y = np.random.choice([0, 1, 2], size=num_samples)
    for i in range(num_samples):
        X[i, :] += y[i] * 2.5 

    # Save to a clean CSV file inside your data/ folder
    columns = [f"feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target_class"] = y
    
    df.to_csv("data/synthetic_data.csv", index=False)
    print("✅ High-dimensional test data successfully saved to 'data/synthetic_data.csv'")

if __name__ == "__main__":
    make_high_dimensional_data()