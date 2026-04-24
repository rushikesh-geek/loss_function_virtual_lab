# =============================================================================
# losses/loss_registry.py
# Maps friendly loss names → instantiated Keras loss objects.
# Centralised here so every model uses the same loss instance.
# =============================================================================

import tensorflow as tf
from losses.focal_loss import FocalLoss


# ── Default hyperparameters (overridden at call time via get_loss) ─────────────
_DEFAULT_HUBER_DELTA  = 1.0
_DEFAULT_FOCAL_ALPHA  = 0.25
_DEFAULT_FOCAL_GAMMA  = 2.0


def get_loss(
    loss_name: str,
    huber_delta: float = _DEFAULT_HUBER_DELTA,
    focal_alpha: float = _DEFAULT_FOCAL_ALPHA,
    focal_gamma: float = _DEFAULT_FOCAL_GAMMA,
):
    """
    Return the appropriate Keras loss object for the given friendly name.

    Parameters
    ----------
    loss_name    : one of the keys below (case-insensitive, spaces OK)
    huber_delta  : threshold δ for Huber loss
    focal_alpha  : α for Focal loss
    focal_gamma  : γ for Focal loss

    Returns
    -------
    tf.keras.losses.Loss instance (ready to pass to model.compile)
    """
    # FIX-2: Normalise to lowercase + stripped before lookup.
    # This handles display names like "MSE Reconstruction", "VAE Loss (Recon + KL)", etc.
    name = loss_name.strip().lower()

    # Also support underscore-separated variants (legacy callers)
    name_underscored = name.replace(" ", "_")

    registry = {
        # ── Multi-class classification ─────────────────────────────────────────
        "cross_entropy":                tf.keras.losses.CategoricalCrossentropy(
                                            from_logits=False,
                                            label_smoothing=0.0,
                                            name="cross_entropy"),
        "cross entropy":                tf.keras.losses.CategoricalCrossentropy(
                                            from_logits=False,
                                            label_smoothing=0.0,
                                            name="cross_entropy"),

        # ── Binary classification ──────────────────────────────────────────────
        "binary_cross_entropy":         tf.keras.losses.BinaryCrossentropy(
                                            from_logits=False,
                                            name="binary_crossentropy"),
        "binary cross entropy":         tf.keras.losses.BinaryCrossentropy(
                                            from_logits=False,
                                            name="binary_crossentropy"),
        "bce":                          tf.keras.losses.BinaryCrossentropy(
                                            from_logits=False,
                                            name="binary_crossentropy"),

        # ── Regression — L2 ───────────────────────────────────────────────────
        "mse":                          tf.keras.losses.MeanSquaredError(name="mse"),

        # ── Robust regression — δ-smooth ──────────────────────────────────────
        "huber":                        tf.keras.losses.Huber(
                                            delta=huber_delta,
                                            name=f"huber_d{huber_delta:.2f}"),

        # ── Imbalanced classification — focal ─────────────────────────────────
        "focal_loss":                   FocalLoss(alpha=focal_alpha, gamma=focal_gamma),
        "focal loss":                   FocalLoss(alpha=focal_alpha, gamma=focal_gamma),

        # ── MAE — included for reference comparisons ──────────────────────────
        "mae":                          tf.keras.losses.MeanAbsoluteError(name="mae"),

        # ── FIX-2: Experiment 5 display-name aliases ──────────────────────────
        # These are the exact strings used as ss.loss_name in Experiment 5.
        # They map to a suitable proxy loss for the loss landscape computation.
        # The VAE itself uses a custom train_step — the registry only needs to
        # return something that satisfies get_loss() without crashing.
        "mse reconstruction":           tf.keras.losses.MeanSquaredError(name="mse"),
        "bce reconstruction":           tf.keras.losses.BinaryCrossentropy(
                                            from_logits=False,
                                            name="binary_crossentropy"),
        "vae loss (recon + kl)":        tf.keras.losses.MeanSquaredError(name="mse"),   # proxy
        "vae loss (recon+kl)":          tf.keras.losses.MeanSquaredError(name="mse"),   # proxy (no spaces)
        "denoising ae (mse)":           tf.keras.losses.MeanSquaredError(name="mse"),
    }

    # Try the normalised name first, then the underscore variant (legacy support)
    if name in registry:
        return registry[name]
    if name_underscored in registry:
        return registry[name_underscored]

    raise ValueError(
        f"Unknown loss '{loss_name}'. "
        f"Available: {sorted(set(registry.keys()))}"
    )


# ── Convenience: list all available loss names ────────────────────────────────
AVAILABLE_LOSSES = [
    "Cross Entropy",
    "Binary Cross Entropy",
    "MSE",
    "Huber",
    "Focal Loss",
    "MAE",
    # Experiment 5 aliases
    "MSE Reconstruction",
    "BCE Reconstruction",
    "VAE Loss (Recon + KL)",
    "Denoising AE (MSE)",
]
