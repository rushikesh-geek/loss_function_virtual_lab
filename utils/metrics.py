# =============================================================================
# utils/metrics.py
# Unified metrics computation for all experiment types.
# Returns a structured dict that app.py can render as a metrics table.
# =============================================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,           # probabilities or class indices or regression values
    task_type: str = "multiclass",  # "multiclass" | "binary" | "regression" | "reconstruction"
    threshold: float = 0.5,
) -> dict:
    """
    Compute all relevant metrics for the given task type.

    Parameters
    ----------
    y_true    : true labels (integer for classification, float for regression)
    y_pred    : model predictions
                - multiclass: (N, K) softmax probabilities
                - binary    : (N,) or (N,1) sigmoid probabilities
                - regression: (N,) continuous values
    task_type : determines which metrics to compute
    threshold : decision threshold for binary classification

    Returns
    -------
    dict with metric names → values (all floats, ready to display)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task_type == "multiclass":
        return _multiclass_metrics(y_true, y_pred)
    elif task_type == "binary":
        return _binary_metrics(y_true, y_pred, threshold)
    elif task_type == "regression":
        return _regression_metrics(y_true, y_pred)
    elif task_type == "reconstruction":
        return _reconstruction_metrics(y_true, y_pred)
    else:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-CLASS
# ─────────────────────────────────────────────────────────────────────────────

def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Metrics for multi-class classification.
    y_pred: (N, K) softmax probabilities or (N,) integer class predictions.
    """
    if y_pred.ndim == 2:
        y_pred_cls = np.argmax(y_pred, axis=1)
    else:
        y_pred_cls = y_pred.astype(int)

    y_true_cls = y_true.astype(int)

    acc   = accuracy_score(y_true_cls, y_pred_cls)
    prec  = precision_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
    rec   = recall_score(y_true_cls, y_pred_cls, average="weighted",    zero_division=0)
    f1    = f1_score(y_true_cls, y_pred_cls, average="weighted",        zero_division=0)

    # AUC (one-vs-rest, if probabilities available)
    auc = None
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        try:
            auc = roc_auc_score(y_true_cls, y_pred, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

    result = {
        "Accuracy":           round(acc,  4),
        "Precision (weighted)": round(prec, 4),
        "Recall (weighted)":  round(rec,  4),
        "F1 (weighted)":      round(f1,   4),
    }
    if auc is not None:
        result["ROC-AUC (OvR)"] = round(auc, 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BINARY
# ─────────────────────────────────────────────────────────────────────────────

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    """
    Metrics for binary classification.
    y_pred: (N,) or (N,1) sigmoid probabilities → thresholded to 0/1.

    ROOT CAUSE 5 FIX:
    -----------------
    sklearn's default pos_label is 1, but by making it EXPLICIT we guarantee:
      • precision = TP_fraud / (TP_fraud + FP_fraud)  [not average over both]
      • recall    = TP_fraud / (TP_fraud + FN_fraud)  [= fraud detection rate]
      • f1        = harmonic mean of above             [0 when fraud undetected]
    This means the metric cards tell students exactly how well fraud is caught,
    not a misleading average that mixes the easy normal class into the score.
    """
    probs     = y_pred.ravel()
    y_true_b  = y_true.ravel().astype(int)
    y_pred_b  = (probs >= threshold).astype(int)

    acc  = accuracy_score(y_true_b, y_pred_b)
    # pos_label=1 → all metrics target the MINORITY class (fraud / positive)
    # zero_division=0 → returns 0.0 (not NaN/warning) when no positives predicted
    prec = precision_score(y_true_b, y_pred_b, pos_label=1, zero_division=0)
    rec  = recall_score(y_true_b,    y_pred_b, pos_label=1, zero_division=0)
    f1   = f1_score(y_true_b,        y_pred_b, pos_label=1, zero_division=0)

    # AUC from continuous probabilities (threshold-independent)
    try:
        auc = roc_auc_score(y_true_b, probs)
    except Exception:
        auc = None

    result = {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4),
        "Threshold": round(threshold, 2),
    }
    if auc is not None:
        result["ROC-AUC"] = round(auc, 4)

    return result


def best_threshold_from_proba(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    pos_label: int = 1,
) -> float:
    """
    ROOT CAUSE 3: Find the decision threshold that maximises F1 on pos_label.

    Uses the Precision-Recall curve, which is more informative than ROC for
    imbalanced datasets.  The optimal point is argmax(F1) over all thresholds.

    Parameters
    ----------
    y_true        : true binary labels (0/1)
    y_pred_proba  : predicted probabilities for the positive class
    pos_label     : which class is the positive (minority) class

    Returns
    -------
    float : optimal threshold in [0, 1], clipped to [0.05, 0.50]
    """
    from sklearn.metrics import precision_recall_curve

    y_t = np.array(y_true).ravel()
    probs = np.array(y_pred_proba).ravel()

    precisions, recalls, thresholds = precision_recall_curve(
        y_t, probs, pos_label=pos_label
    )
    # thresholds has len N-1; precisions/recalls have len N (last entry = trivial)
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )
    if len(f1_scores) == 0 or f1_scores.max() == 0:
        return 0.20   # sensible fallback

    best_idx = int(np.argmax(f1_scores))
    best_thr = float(thresholds[best_idx])
    return float(np.clip(best_thr, 0.05, 0.50))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for regression: MSE, RMSE, MAE, R²."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {
        "MSE":  round(mse,  4),
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "R²":   round(r2,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for autoencoder reconstruction."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    # PSNR: Peak Signal-to-Noise Ratio (higher = better reconstruction)
    psnr = max(0.0, 10 * np.log10(1.0 / (mse + 1e-12)))

    return {
        "MSE (pixel)":  round(mse,  6),
        "MAE (pixel)":  round(mae,  6),
        "PSNR (dB)":    round(psnr, 2),
    }
