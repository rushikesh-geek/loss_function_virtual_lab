# =============================================================================
# models/mlp.py
# Multi-Layer Perceptron builder for classification and regression tasks.
# Architecture is configurable via parameters — no hardcoded sizes.
# =============================================================================

import tensorflow as tf
from tensorflow import keras


def build_mlp(
    input_dim: int,
    output_dim: int,
    task_type: str = "multiclass",   # "multiclass" | "binary" | "regression"
    hidden_units: tuple = (256, 128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    optimizer_name: str = "Adam",
    loss_name: str = "Cross Entropy",   # friendly name from loss_registry
    # Focal Loss specific params (passed through to loss_registry)
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    # Huber specific
    huber_delta: float = 1.0,
) -> keras.Model:
    """
    Build and compile a fully-connected MLP.

    Architecture
    ────────────
    Input → [Dense(h) → BatchNorm → ReLU → Dropout] × len(hidden) → Output

    Parameters
    ----------
    input_dim      : number of input features
    output_dim     : number of output units
    task_type      : determines output activation + default metric
    hidden_units   : tuple of hidden layer sizes
    dropout_rate   : dropout fraction after each hidden layer
    learning_rate  : optimizer learning rate
    optimizer_name : "Adam" | "SGD" | "RMSprop"
    loss_name      : friendly loss name (resolved by loss_registry)

    Returns
    -------
    Compiled Keras Model
    """
    from losses.loss_registry import get_loss

    # ── Resolve loss and output activation ────────────────────────────────────
    loss_fn     = get_loss(loss_name,
                           huber_delta=huber_delta,
                           focal_alpha=focal_alpha,
                           focal_gamma=focal_gamma)
    out_act, metrics = _get_output_config(task_type)

    # ── Build model ───────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(input_dim,), name="features")
    x      = inputs

    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, name=f"dense_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = keras.layers.Dense(output_dim, activation=out_act, name="output")(x)
    model   = keras.Model(inputs=inputs, outputs=outputs, name="MLP")

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = _build_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model


def _get_output_config(task_type: str):
    """Map task type → (output_activation, metrics_list)."""
    configs = {
        "multiclass":  ("softmax", ["accuracy"]),
        "binary":      ("sigmoid", ["accuracy",
                                    tf.keras.metrics.AUC(name="auc",
                                                          curve="ROC")]),
        "regression":  (None,      ["mae"]),
    }
    return configs.get(task_type, ("softmax", ["accuracy"]))


def _build_optimizer(name: str, lr: float):
    """Instantiate Keras optimizer by friendly name."""
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    else:
        return keras.optimizers.Adam(learning_rate=lr)
