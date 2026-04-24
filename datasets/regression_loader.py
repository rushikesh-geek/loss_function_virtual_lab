# =============================================================================
# datasets/regression_loader.py
# Loads California Housing dataset + synthetic outlier injection.
# Used by Experiment 3 (Regression + Outlier Robustness).
# =============================================================================

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_california_housing_data(
    n_outliers: int = 0,
    outlier_magnitude: float = 5.0,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load California Housing dataset with optional outlier injection.

    California Housing (sklearn):
        - 20,640 samples, 8 features
        - Target: median house value (in $100,000), clipped to [0.15, 5.0]

    Parameters
    ----------
    n_outliers         : number of training samples to replace with extreme outliers
    outlier_magnitude  : multiplier for outlier y-values (e.g., 5.0 = 5× max value)
    test_size          : fraction of data used for testing
    random_state       : reproducibility seed

    Returns
    -------
    dict with:
        X_train, X_test : scaled feature arrays
        y_train, y_test : target values (house price in $100k)
        y_train_clean   : original y_train without outliers (for reference)
        feature_names   : list of 8 feature name strings
        n_outliers      : how many outliers were injected
    """
    housing = fetch_california_housing()
    X, y    = housing.data, housing.target
    feature_names = list(housing.feature_names)

    # ── Subsample for speed (keep 5000 training samples max) ─────────────────
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), min(5000, len(X)), replace=False)
    X, y = X[idx], y[idx]

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── Standardise features ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test  = scaler.transform(X_test).astype("float32")

    y_train = y_train.astype("float32")
    y_test  = y_test.astype("float32")
    y_train_clean = y_train.copy()

    # ── Inject synthetic outliers ────────────────────────────────────────────
    # Replace random training labels with extreme values to simulate
    # heavy-tailed noise (e.g., data entry errors, sensor malfunctions).
    if n_outliers > 0:
        n_outliers = min(n_outliers, len(y_train) - 10)   # safety cap
        outlier_idx = rng.choice(len(y_train), n_outliers, replace=False)
        y_max = float(y_train.max())
        # Alternate between positive and negative outliers for variety
        signs = rng.choice([-1, 1], n_outliers)
        y_train[outlier_idx] = signs * y_max * outlier_magnitude

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "y_train_clean": y_train_clean,
        "feature_names": feature_names,
        "n_features":    X_train.shape[1],
        "n_outliers":    n_outliers,
    }
