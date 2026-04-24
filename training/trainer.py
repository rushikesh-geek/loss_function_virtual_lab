# =============================================================================
# training/trainer.py
# Unified Keras training interface with:
#   • Epoch progress callbacks (updates Streamlit progress bar)
#   • Gradient magnitude recording per layer per epoch
#   • History capture for all metrics
#   • Comparison mode (train 2 losses, return both histories)
#
# BUG FIX (TypeError: 'str' object is not callable)
# -------------------------------------------------
# The Keras model.loss attribute is a STRING like "mse" or "binary_crossentropy"
# when the model is compiled with a string loss name.  GradientRecorder then
# tries to call self.loss_fn(y_true, y_pred) → TypeError crash.
#
# Fix: resolve string → callable at __init__ time via LOSS_CALLABLE_MAP.
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Callable


# ─────────────────────────────────────────────────────────────────────────────
# STRING → CALLABLE LOSS RESOLVER  (BUG FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_loss_callable(loss_fn) -> Optional[Callable]:
    """
    Resolve a Keras loss to a Python callable.

    If loss_fn is already callable, return it directly.
    If it is a string (e.g. "mse", "binary_crossentropy"), look it up in
    LOSS_CALLABLE_MAP and return the corresponding Keras loss object.
    Returns None if the loss cannot be resolved (e.g. VAE custom train_step
    models where model.loss is None or not reliably callable).

    This prevents the TypeError: 'str' object is not callable crash in
    GradientRecorder.on_epoch_end().
    """
    if loss_fn is None:
        return None

    if callable(loss_fn):
        return loss_fn

    # Lazily import to avoid circular import at module level
    from losses.focal_loss import FocalLoss

    LOSS_CALLABLE_MAP = {
        # Regression / reconstruction
        "mse":                      tf.keras.losses.MeanSquaredError(),
        "mean_squared_error":       tf.keras.losses.MeanSquaredError(),
        "mae":                      tf.keras.losses.MeanAbsoluteError(),
        "mean_absolute_error":      tf.keras.losses.MeanAbsoluteError(),
        "huber":                    tf.keras.losses.Huber(),
        "huber_loss":               tf.keras.losses.Huber(),
        # Classification
        "binary_crossentropy":      tf.keras.losses.BinaryCrossentropy(),
        "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
        "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(),
        # Custom
        "focal":                    FocalLoss(),
        "focal_loss":               FocalLoss(),
    }

    if isinstance(loss_fn, str):
        resolved = LOSS_CALLABLE_MAP.get(loss_fn.lower())
        if resolved is not None:
            return resolved
        # Unknown string — return MSE as a safe fallback
        return tf.keras.losses.MeanSquaredError()

    return None   # completely unresolvable — caller will skip gradient recording


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback(keras.callbacks.Callback):
    """
    Update a Streamlit progress bar and status text each epoch.

    Parameters
    ----------
    progress_fn : callable(fraction: float) — receives value in [0,1]
    status_fn   : callable(text: str)       — displays epoch/loss text
    total_epochs: int
    """
    def __init__(self, progress_fn: Callable, status_fn: Callable, total_epochs: int):
        super().__init__()
        self.progress_fn  = progress_fn
        self.status_fn    = status_fn
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        fraction = (epoch + 1) / self.total_epochs
        self.progress_fn(fraction)

        # Support multiple named loss keys (VAE logs total_loss/recon_loss/kl_loss)
        loss_val = logs.get("loss", logs.get("total_loss", 0.0))
        val_val  = logs.get("val_loss", logs.get("val_total_loss", None))
        acc_val  = logs.get("accuracy", logs.get("mae", None))

        status = f"Epoch {epoch+1}/{self.total_epochs} — Loss: {loss_val:.4f}"
        if val_val is not None:
            status += f" | Val Loss: {val_val:.4f}"
        if acc_val is not None:
            metric_name = "Acc" if "accuracy" in logs else "MAE"
            status += f" | {metric_name}: {acc_val:.4f}"

        # Extra VAE info
        kl = logs.get("kl_loss", None)
        rc = logs.get("recon_loss", None)
        if kl is not None and rc is not None:
            status += f" | Recon: {rc:.2f} | KL: {kl:.2f}"

        self.status_fn(status)


class GradientRecorder(keras.callbacks.Callback):
    """
    Record gradient magnitude (L2 norm) per trainable layer per epoch.

    After training, access via `self.gradient_history`:
        dict[layer_name: str] → list[float]  (length = n_epochs)

    Note: Uses a sample batch from training data, not the full dataset.
    This is intentionally lightweight for interactive use.

    BUG FIX: loss_fn is resolved to a callable in __init__ via
    _resolve_loss_callable() so on_epoch_end never crashes with
    "TypeError: 'str' object is not callable".
    """
    def __init__(self, X_sample, y_sample, loss_fn, max_layers: int = 6):
        super().__init__()
        self.X_sample   = tf.constant(X_sample[:128], dtype=tf.float32)
        self.y_sample   = tf.constant(y_sample[:128], dtype=tf.float32)
        # ── BUG FIX: resolve string → callable here, once, safely ─────────
        self.loss_fn    = _resolve_loss_callable(loss_fn)
        self.max_layers = max_layers       # limit to avoid huge legends
        self.gradient_history = {}         # layer_name → [grad_norm_epoch0, ...]

    def on_epoch_end(self, epoch: int, logs: dict = None):
        # Skip if we couldn't resolve a callable (e.g. VAE custom train_step)
        if self.loss_fn is None:
            return

        # Compute gradients w.r.t. trainable variables
        try:
            with tf.GradientTape() as tape:
                preds = self.model(self.X_sample, training=False)
                loss  = self.loss_fn(self.y_sample, preds)
        except Exception:
            # If gradient computation fails for any model (e.g. VAE with multi-output),
            # silently skip this epoch rather than crashing the training loop.
            return

        grads = tape.gradient(loss, self.model.trainable_variables)

        # Record L2 norm (Frobenius norm for matrices) per variable
        recorded = 0
        seen_layers = set()
        for var, grad in zip(self.model.trainable_variables, grads):
            if grad is None:
                continue
            # Extract base layer name (strip weight suffix)
            layer_name = var.name.split("/")[0]
            if layer_name in seen_layers:
                continue              # only first var per layer (weights, not bias)
            seen_layers.add(layer_name)

            norm = float(tf.norm(grad).numpy())
            if layer_name not in self.gradient_history:
                self.gradient_history[layer_name] = []
            self.gradient_history[layer_name].append(norm)

            recorded += 1
            if recorded >= self.max_layers:
                break


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Unified training interface for all experiments.

    Usage
    -----
    trainer = Trainer(model, X_train, y_train)
    history, grad_history = trainer.train(
        epochs=10, batch_size=64,
        progress_fn=st.progress,
        status_fn=st.write,
    )

    For VAE models (which use a custom train_step and should NOT be compiled
    with a loss argument), pass record_gradients=False to skip GradientRecorder.
    The Trainer will still call model.fit() correctly.
    """

    def __init__(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.15,
    ):
        self.model            = model
        self.X_train          = X_train
        self.y_train          = y_train
        self.validation_split = validation_split

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        progress_fn: Optional[Callable] = None,
        status_fn: Optional[Callable]   = None,
        record_gradients: bool          = True,
        class_weight: Optional[dict]    = None,   # ROOT CAUSE 1 — imbalance fix
    ) -> tuple:
        """
        Run model.fit() with optional progress callbacks.

        Parameters
        ----------
        epochs           : training epochs
        batch_size       : mini-batch size
        progress_fn      : callable(float) — Streamlit progress bar update
        status_fn        : callable(str)   — status text update
        record_gradients : whether to run GradientRecorder callback.
                           Set to False for VAE (custom train_step) models.
        class_weight     : optional dict {class_index: weight} passed directly
                           to model.fit().  When provided, Keras multiplies each
                           sample's loss by its class weight, so minority-class
                           samples contribute proportionally more gradient update.
                           This is the SIMPLEST and most reliable fix for silent
                           model collapse on imbalanced datasets.

        Returns
        -------
        (history_dict, gradient_history_dict)
        history_dict       : {"loss": [...], "val_loss": [...], "accuracy": [...], ...}
        gradient_history   : {"layer_name": [norm_e0, norm_e1, ...], ...}
        """
        callbacks = []

        # ── Progress callback ─────────────────────────────────────────────────
        if progress_fn is not None and status_fn is not None:
            callbacks.append(
                ProgressCallback(progress_fn, status_fn, epochs)
            )

        # ── Gradient recorder ─────────────────────────────────────────────────
        # BUG FIX: Use _resolve_loss_callable() to safely turn the model's
        # loss attribute (which may be a plain string like "mse") into a
        # callable before passing to GradientRecorder.
        grad_recorder = None
        if record_gradients:
            raw_loss_fn   = self.model.loss          # may be str, callable, or None
            resolved_loss = _resolve_loss_callable(raw_loss_fn)
            if resolved_loss is not None:
                grad_recorder = GradientRecorder(
                    self.X_train, self.y_train, resolved_loss
                )
                callbacks.append(grad_recorder)
            # else: skip gradient recording (e.g. VAE whose model.loss is None)

        # FIX: Detect VAE by checking for encoder, decoder AND total_loss_tracker.
        # total_loss_tracker is the keras.metrics.Mean() object that the VAE's
        # custom train_step populates — it is the definitive VAE signal.
        # A standard autoencoder compiled with loss="mse" has encoder+decoder
        # but NO total_loss_tracker, so it follows the normal "val_loss" path.
        is_vae = (
            hasattr(self.model, "encoder") and          # FIX: Change 1 — tighter VAE check
            hasattr(self.model, "decoder") and
            hasattr(self.model, "total_loss_tracker")   # FIX: only the custom-train_step VAE has this
        )
        val_monitor = "val_total_loss" if is_vae else "val_loss"

        # ── Learning rate schedule ─────────────────────────────────────────────
        # FIX: Change 2 — pass mode='min' EXPLICITLY so Keras never tries to
        # auto-detect direction from the custom metric name "val_total_loss".
        # Keras auto-detection only works for a short whitelist of built-in names
        # ("loss", "val_loss", "accuracy", "val_accuracy") — any custom name like
        # "val_total_loss" raises ValueError: "Keras isn't able to automatically
        # determine whether that metric should be maximized or minimized."
        # All loss metrics should be minimized, so mode='min' is always correct.
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor  = val_monitor,
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            mode     = 'min',   # FIX: Change 2 — always explicit, never 'auto'
            verbose  = 0,
        )
        callbacks.append(lr_scheduler)

        # ── Early stopping ─────────────────────────────────────────────────────
        # FIX: Change 2 (continued) — same explicit mode='min' required here.
        early_stop = keras.callbacks.EarlyStopping(
            monitor              = val_monitor,
            patience             = max(epochs // 3, 3),
            restore_best_weights = True,
            mode                 = 'min',   # FIX: Change 2 — always explicit
            verbose              = 0,
        )
        callbacks.append(early_stop)

        # ── Run training ───────────────────────────────────────────────────────
        # FIX: Change 3 — set fit_y and val_data correctly for VAE vs non-VAE.
        # For the VAE: target = input (reconstruction task).
        #   model.fit(X, X, ...) creates (x_batch, x_batch) pairs per step.
        #   VAE.train_step unpacks as (x, _) — it ignores the y column
        #   and computes the ELBO internally. Passing y_train here would
        #   cause a shape mismatch because y_train has different dims.
        # For all other models: target = y_train as normal.
        if is_vae:                              # FIX: Change 3
            fit_y = self.X_train               # FIX: reconstruction target = input
        else:
            fit_y = self.y_train

        history = self.model.fit(
            self.X_train, fit_y,
            epochs           = epochs,
            batch_size       = batch_size,
            validation_split = self.validation_split,
            callbacks        = callbacks,
            class_weight     = class_weight,
            verbose          = 0,
        )

        # ── Package history as plain dict ─────────────────────────────────────
        history_dict     = {k: list(v) for k, v in history.history.items()}
        gradient_history = grad_recorder.gradient_history if grad_recorder else {}

        return history_dict, gradient_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model predictions as numpy array."""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model and return metrics dict.
        Keys depend on what was compiled (accuracy, mae, auc, etc.).
        """
        results  = self.model.evaluate(X, y, verbose=0)
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))
