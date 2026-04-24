# =============================================================================
# plots/roc_curve.py
# ROC Curve and Precision-Recall Curve via Plotly.
# Works for binary and multi-class (one-vs-rest) cases.
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from config import COLORS, PLOTLY_TEMPLATE, CHART_HEIGHT, PLOTLY_COLORS


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list = None,
    title: str = "ROC Curve",
) -> go.Figure:
    """
    Plot ROC curve(s) with AUC scores.

    For binary   : single ROC curve.
    For multi-class: one-vs-rest micro/macro averaged curves.

    Parameters
    ----------
    y_true       : true integer labels
    y_pred_probs : predicted probabilities — (N,) for binary, (N,K) for multi-class
    class_names  : list of class name strings
    title        : chart title

    Returns
    -------
    plotly.graph_objects.Figure with hover tooltips
    """
    fig  = go.Figure()
    y_t  = np.array(y_true).ravel()

    # ── Binary case ───────────────────────────────────────────────────────────
    if y_pred_probs.ndim == 1 or (y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 1):
        probs = y_pred_probs.ravel()
        fpr, tpr, thresholds = roc_curve(y_t, probs)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(74,144,217,0.1)",
            name=f"ROC (AUC = {roc_auc:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))

    # ── Multi-class (one-vs-rest) ─────────────────────────────────────────────
    else:
        n_classes = y_pred_probs.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        for i in range(n_classes):
            y_bin = (y_t == i).astype(int)
            probs_i = y_pred_probs[:, i]
            fpr, tpr, _ = roc_curve(y_bin, probs_i)
            roc_auc = auc(fpr, tpr)
            color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color=color, width=1.8),
                name=f"Class {class_names[i]} (AUC={roc_auc:.2f})",
                hovertemplate=f"Class {class_names[i]}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
            ))

    # ── Diagonal (random classifier) ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", width=1.5, dash="dash"),
        name="Random (AUC = 0.5)",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        xaxis=dict(title="False Positive Rate", range=[0, 1], gridcolor="#2D3250"),
        yaxis=dict(title="True Positive Rate",  range=[0, 1.02], gridcolor="#2D3250"),
        height=CHART_HEIGHT,
        legend=dict(font=dict(size=11)),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list = None,
    title: str = "Precision-Recall Curve",
) -> go.Figure:
    """
    Plot Precision-Recall curve.
    Particularly important for imbalanced datasets (Experiment 4).
    """
    fig = go.Figure()
    y_t = np.array(y_true).ravel()

    # ── Binary ────────────────────────────────────────────────────────────────
    if y_pred_probs.ndim == 1 or (y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 1):
        probs = y_pred_probs.ravel()
        precision, recall, _ = precision_recall_curve(y_t, probs)
        ap = average_precision_score(y_t, probs)

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            line=dict(color=COLORS["loss_accent"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.1)",
            name=f"PR Curve (AP = {ap:.3f})",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))

        # Baseline: random classifier (= fraud_rate)
        baseline = y_t.mean()
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline (= {baseline:.3f})",
            annotation_position="bottom right",
        )

    # ── Multi-class ───────────────────────────────────────────────────────────
    else:
        n_classes = y_pred_probs.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        for i in range(n_classes):
            y_bin       = (y_t == i).astype(int)
            probs_i     = y_pred_probs[:, i]
            prec, rec, _= precision_recall_curve(y_bin, probs_i)
            ap          = average_precision_score(y_bin, probs_i)
            color       = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]

            fig.add_trace(go.Scatter(
                x=rec, y=prec,
                mode="lines",
                line=dict(color=color, width=1.8),
                name=f"Class {class_names[i]} (AP={ap:.2f})",
                hovertemplate=(
                    f"Class {class_names[i]}<br>"
                    "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        xaxis=dict(title="Recall",    range=[0, 1], gridcolor="#2D3250"),
        yaxis=dict(title="Precision", range=[0, 1.02], gridcolor="#2D3250"),
        height=CHART_HEIGHT,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig
