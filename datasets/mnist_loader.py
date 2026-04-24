# =============================================================================
# datasets/mnist_loader.py
# Loads MNIST from keras.datasets, preprocesses, and returns numpy arrays.
# Used by Experiments 1 (classification) and 5 (autoencoder).
# =============================================================================

import numpy as np
from tensorflow import keras


def load_mnist(flatten: bool = True, subset: int = 10000):
    """
    Load and preprocess MNIST dataset.

    Parameters
    ----------
    flatten : bool
        If True, images are shape (N, 784). If False, shape (N, 28, 28, 1).
        Use flatten=True for MLP, flatten=False for CNN/Autoencoder.
    subset : int
        Number of training samples to use (for speed in the interactive lab).
        Test set always uses 2000 samples.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test,
        X_train_flat, X_test_flat   (always available),
        y_train_cat, y_test_cat     (one-hot encoded),
        n_classes, input_shape
    """
    # Load raw MNIST (downloads once, cached by Keras)
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

    # ── Subsample for speed ──────────────────────────────────────────────────
    idx_train = np.random.choice(len(X_train_raw), min(subset, len(X_train_raw)), replace=False)
    idx_test  = np.random.choice(len(X_test_raw),  min(2000, len(X_test_raw)),   replace=False)

    X_train_raw = X_train_raw[idx_train]
    y_train_raw = y_train_raw[idx_train]
    X_test_raw  = X_test_raw[idx_test]
    y_test_raw  = y_test_raw[idx_test]

    # ── Normalise to [0, 1] ──────────────────────────────────────────────────
    X_train_4d = X_train_raw.astype("float32") / 255.0
    X_test_4d  = X_test_raw.astype("float32")  / 255.0

    # Add channel dimension: (N, 28, 28) → (N, 28, 28, 1)
    X_train_4d = X_train_4d[..., np.newaxis]
    X_test_4d  = X_test_4d[..., np.newaxis]

    # Flat versions for MLP
    X_train_flat = X_train_4d.reshape(len(X_train_4d), -1)
    X_test_flat  = X_test_4d.reshape(len(X_test_4d),   -1)

    # One-hot encode labels for CE loss
    n_classes    = 10
    y_train_cat  = keras.utils.to_categorical(y_train_raw, n_classes)
    y_test_cat   = keras.utils.to_categorical(y_test_raw,  n_classes)

    return {
        "X_train":       X_train_4d,      # (N, 28, 28, 1)  — for CNN/AE
        "X_test":        X_test_4d,
        "X_train_flat":  X_train_flat,    # (N, 784)         — for MLP
        "X_test_flat":   X_test_flat,
        "y_train":       y_train_raw,     # (N,) integer labels
        "y_test":        y_test_raw,
        "y_train_cat":   y_train_cat,     # (N, 10) one-hot
        "y_test_cat":    y_test_cat,
        "n_classes":     n_classes,
        "input_shape_flat": (784,),
        "input_shape_2d":   (28, 28, 1),
    }
