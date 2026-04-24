# =============================================================================
# losses/focal_loss.py
# Custom Focal Loss implementation inheriting from tf.keras.losses.Loss.
# Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
#            https://arxiv.org/abs/1708.02002
#
# Key idea: down-weight easy examples so model focuses on hard minority class.
#   FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
#   • gamma=0  → standard Binary Cross Entropy
#   • gamma=2  → the paper's default; easy examples 100× down-weighted
#   • alpha    → compensates for class imbalance
#
# ROOT CAUSE 2 FIX:
#   Old default alpha = 0.25 was designed for COCO object detection where the
#   positive class is ~15% of anchors.  For 5% fraud, the minority class needs
#   alpha ~ 0.95 (= 1 - fraud_ratio) so loss contribution from fraud examples
#   is up-weighted relative to normal examples.
#   The actual value is set dynamically in _train_exp4() based on fraud_ratio.
# =============================================================================

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for imbalanced classification.

    Parameters
    ----------
    alpha : float
        Weighting factor in [0,1] for the MINORITY (positive / fraud) class.
        RULE OF THUMB: set alpha ≈ 1 - fraud_ratio.
        • If fraud is 5%  → alpha ≈ 0.95  (fraud samples get 19× more weight)
        • If fraud is 10% → alpha ≈ 0.90
        • alpha = 0.25 (old default) would UNDER-weight the minority class for
          small fraud ratios and is a primary reason for precision/recall = 0.

    gamma : float
        Focusing parameter ≥ 0. Controls how fast easy examples are down-weighted.
        gamma = 0 → standard BCE (no focusing).
        gamma = 2 → default from paper; correct easy examples get 100× less loss.
        gamma = 4 → even stronger focus on hard examples (use if still collapsing).

    Example
    -------
    >>> fl = FocalLoss(alpha=0.95, gamma=2.0)   # for 5% fraud ratio
    >>> loss = fl(y_true, y_pred)
    """

    def __init__(
        self,
        alpha: float = 0.95,    # ← ROOT CAUSE 2: changed from 0.25 to 0.95
        gamma: float = 2.0,
        reduction: str = "sum_over_batch_size",
        name: str = "focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    @tf.function   # JIT-compile for speed in training loop
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss for a batch.

        Parameters
        ----------
        y_true : tf.Tensor of shape (batch,) — binary labels {0, 1}
        y_pred : tf.Tensor of shape (batch,) — predicted probabilities in [0,1]

        Returns
        -------
        Per-example focal loss values (before reduction)

        How it works
        ------------
        Standard BCE already measures the prediction error, but it gives EQUAL
        weight to easy (already-correct) and hard (currently-wrong) examples.
        Focal loss multiplies BCE by (1 - p_t)^gamma, which:
          • approaches 0  for easy correct predictions  (p_t → 1)
          • stays near 1  for hard wrong predictions    (p_t → 0)
        Combined with alpha weighting that up-weights the minority class,
        the model is forced to focus gradient updates on fraud samples.
        """
        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to avoid log(0) / log(1) = -inf / 0
        epsilon = tf.keras.backend.epsilon()    # 1e-7
        y_pred  = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # ── Binary Cross-Entropy component ────────────────────────────────────
        # bce = -[y·log(p) + (1-y)·log(1-p)]
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        # ── p_t: probability assigned to the CORRECT class ───────────────────
        # p_t = p    if y_true = 1  (model should predict HIGH; p_t = predicted prob)
        # p_t = 1-p  if y_true = 0  (model should predict LOW;  p_t = 1 - predicted)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # ── Modulating factor: (1 - p_t)^gamma ───────────────────────────────
        # When p_t → 1 (easy, correct prediction): factor → 0  (down-weighted)
        # When p_t → 0 (hard, wrong prediction):   factor → 1  (full weight)
        # gamma=2 means a correctly-predicted easy sample (p_t=0.9) contributes
        # only (1-0.9)^2 = 0.01 of its BCE — 100× less than a hard sample.
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # ── Alpha weighting (ROOT CAUSE 2 fix) ───────────────────────────────
        # alpha_t = alpha       for positive class (y=1, fraud)
        # alpha_t = 1 - alpha   for negative class (y=0, normal)
        # With alpha=0.95:  fraud samples get 0.95 weight,
        #                   normal samples get 0.05 weight.
        # This is the key multiplier that stops the loss collapsing to "predict
        # all normal" — normal samples simply don't contribute much gradient.
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        # ── Final focal loss ─────────────────────────────────────────────────
        focal_loss = alpha_t * modulating_factor * bce

        return focal_loss

    def get_config(self) -> dict:
        """Needed for model serialisation (save/load)."""
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config
