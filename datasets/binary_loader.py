# =============================================================================
# datasets/binary_loader.py
# Loads Breast Cancer Wisconsin dataset from sklearn.
# Used by Experiment 2 (Binary Classification).
# =============================================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_breast_cancer_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load and preprocess the Breast Cancer Wisconsin dataset.

    Features: 30 numeric features (radius, texture, perimeter, area, etc.)
    Target  : 0 = malignant, 1 = benign
    Size    : 569 samples

    Parameters
    ----------
    test_size    : fraction of data used for testing
    random_state : reproducibility seed

    Returns
    -------
    dict with:
        X_train, X_test : scaled feature arrays
        y_train, y_test : binary labels (0/1)
        feature_names   : list of 30 feature name strings
        class_names     : ['malignant', 'benign']
        n_features      : 30
    """
    data = load_breast_cancer()
    X, y = data.data, data.target          # y: 0=malignant, 1=benign
    feature_names = list(data.feature_names)
    class_names   = list(data.target_names)  # ['malignant', 'benign']

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Standardise (zero mean, unit variance) ───────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test  = scaler.transform(X_test).astype("float32")

    # Cast labels to float32 for Keras binary CE
    y_train = y_train.astype("float32")
    y_test  = y_test.astype("float32")

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": feature_names,
        "class_names":   class_names,
        "n_features":    X_train.shape[1],
    }
