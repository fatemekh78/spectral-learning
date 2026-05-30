# Spectral Learning Framework from Scratch

A comprehensive, production-grade implementation of **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)** built strictly from scratch using fundamental linear algebra operations. This framework includes data scaling, variance evaluation, feature loading analysis, and side-by-side benchmark comparisons with non-linear manifold learning (UMAP).

---

## 🎯 Project Objectives
* **Mathematical Authenticity:** Implement core spectral decomposition algorithms without relying on high-level libraries like `scikit-learn` or `scipy.linalg.svds`.
* **Reusability:** Design a modular math matrix engine utilized uniformly across both dimensionality reduction pipelines.
* **Feature Interpretability:** Extract and visualize latent components to map high-dimensional information back to raw feature spaces.
* **Comparative Visual Analysis:** Benchmark linear matrix factorization strategies directly against non-linear architectures.

---

## 📂 Project Architecture
```text
├── data/                       # Contains raw datasets and generated visualization assets
├── docs/                       # Detailed mathematical descriptions and algorithm theory
├── models/
│   ├── pca.py                  # Object-oriented, from-scratch Custom PCA model
│   └── svd.py                  # Object-oriented, from-scratch Custom SVD model
├── utils/
│   ├── data_loader.py          # Strict data cleaning, handling nulls, and custom scaling layer
│   ├── matrix_ops.py           # Reusable mathematical backend (sorted eigen-decomposition)
│   └── visualizations.py       # High-resolution plotting engine and metric generators
├── run_visualizations.py       # Execution master script for graphic generation
├── run_evaluation.py           # Execution master script for quantitative metrics
└── requirements.txt            # System dependencies