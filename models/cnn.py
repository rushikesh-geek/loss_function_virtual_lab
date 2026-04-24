# =============================================================================
# models/cnn.py  — FIXED VERSION
# Small Convolutional Neural Network for MNIST multi-class classification.
# Experiment 1: Compare Cross-Entropy / MSE / Focal / Label-Smoothing on MNIST.
#
# ROOT-CAUSE FIX:  Removed BatchNormalization layers.
#   BatchNorm stores running mean/variance updated only during training=True.
#   During inference (training=False) it uses those accumulated statistics.
#   In Keras 3.x with short training or small batches the running stats can be
#   poorly initialised, causing a massive train/test accuracy gap and the model
#   collapsing to predict a single class at test-time.
#   Solution: use a clean Conv → MaxPool → Dense architecture without BN.
#
# Architecture (exactly as specified):
#   Input (28,28,1)
#   Conv2D(32, 3×3, relu, same)
#   MaxPooling2D(2,2)            → 14×14×32
#   Conv2D(64, 3×3, relu, same)
#   MaxPooling2D(2,2)            → 7×7×64
#   Flatten                      → 3136
#   Dense(128, relu)
#   Dropout(rate)
#   Dense(10, softmax)           → class probabilities
# =============================================================================

import tensorflow as tf
from tensorflow import keras


def build_cnn(
    input_shape: tuple = (28, 28, 1),
    n_classes: int = 10,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    optimizer_name: str = "Adam",
    loss_name: str = "Cross Entropy",
) -> keras.Model:
    """
    Build and compile a clean Small CNN for MNIST classification.

    FIX vs original: BatchNormalization REMOVED to prevent train/test mismatch.
    BatchNorm with short training in Keras 3.x leads to poorly-calibrated
    running statistics → model collapses to one class during inference.

    Architecture
    ────────────
    Conv2D(32, 3×3, relu) → MaxPool(2×2)
    Conv2D(64, 3×3, relu) → MaxPool(2×2)
    Flatten → Dense(128, relu) → Dropout → Dense(10, softmax)

    Parameters
    ----------
    input_shape    : (H, W, C) — (28, 28, 1) for MNIST
    n_classes      : number of output classes (10 for MNIST digits 0-9)
    dropout_rate   : dropout fraction before final dense layer
    learning_rate  : optimizer learning rate
    optimizer_name : "Adam" | "SGD" | "RMSprop"
    loss_name      : friendly loss name (resolved via loss_registry)

    Returns
    -------
    Compiled keras.Model
    """
    from losses.loss_registry import get_loss

    loss_fn   = get_loss(loss_name)
    optimizer = _build_optimizer(optimizer_name, learning_rate)

    # ── Build model using Functional API (more explicit than Sequential) ───────
    # Using keras.Input makes the data-flow graph unambiguous and avoids
    # the implicit input-shape issues in the Keras 3 Sequential API.
    inputs = keras.Input(shape=input_shape, name="image_input")

    # ── Block 1: low-level edge/texture detection ─────────────────────────────
    x = keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", name="conv1"
    )(inputs)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)      # 28→14

    # ── Block 2: mid-level shape detection ───────────────────────────────────
    x = keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name="conv2"
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)      # 14→7

    # ── Classifier head ───────────────────────────────────────────────────────
    x = keras.layers.Flatten(name="flatten")(x)                 # 7×7×64 = 3136
    x = keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)

    # Output: 10-way softmax → probability distribution over digits 0-9
    outputs = keras.layers.Dense(
        n_classes, activation="softmax", name="output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SmallCNN")

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        # "accuracy" resolves to CategoricalAccuracy when labels are one-hot
        # or SparseCategoricalAccuracy for integer labels — Keras 3 handles both
        metrics=["accuracy"],
    )
    return model


# ── MLP variant for flat-input experiments ───────────────────────────────────
def build_mlp_for_mnist(
    input_dim: int = 784,
    n_classes: int = 10,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    optimizer_name: str = "Adam",
    loss_name: str = "Cross Entropy",
) -> keras.Model:
    """
    MLP for MNIST (no BatchNorm — same reason as CNN above).

    Architecture: Dense(512,relu) → Dropout → Dense(256,relu) → Dropout → Dense(10,softmax)
    """
    from losses.loss_registry import get_loss

    loss_fn   = get_loss(loss_name)
    optimizer = _build_optimizer(optimizer_name, learning_rate)

    inputs  = keras.Input(shape=(input_dim,), name="flat_input")
    x       = keras.layers.Dense(512, activation="relu", name="fc1")(inputs)
    x       = keras.layers.Dropout(dropout_rate, name="drop1")(x)
    x       = keras.layers.Dense(256, activation="relu", name="fc2")(x)
    x       = keras.layers.Dropout(dropout_rate, name="drop2")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_MNIST")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model


def _build_optimizer(name: str, lr: float):
    """Instantiate Keras optimizer by friendly name."""
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        # Momentum makes SGD competitive with Adam on MNIST
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    return keras.optimizers.Adam(learning_rate=lr)
