# =============================================================================
# plots/confusion_matrix.py
# Interactive Plotly confusion matrix heatmap.
# Supports binary (2×2) and multi-class (N×N) cases.
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from config import COLORS, PLOTLY_TEMPLATE


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    threshold: float = 0.5,    # used only for binary (y_pred as probabilities)
    normalize: bool = True,
) -> go.Figure:
    """
    Generate an interactive Plotly confusion matrix heatmap.

    Parameters
    ----------
    y_true       : true integer labels (0, 1, ..., N-1) or binary floats
    y_pred       : predicted probabilities or integer class indices
    class_names  : list of class label strings; auto-generated if None
    title        : chart title
    threshold    : decision threshold for binary classification
    normalize    : show counts as percentages (row-normalised)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from sklearn.metrics import confusion_matrix

    # ── Convert probabilities → class indices ─────────────────────────────────
    y_true = np.array(y_true).ravel()

    if y_pred.ndim == 1:
        # Binary sigmoid output → apply threshold
        y_pred_cls = (y_pred >= threshold).astype(int)
    elif y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred_cls = (y_pred.ravel() >= threshold).astype(int)
    elif y_pred.ndim == 2:
        y_pred_cls = np.argmax(y_pred, axis=1)
    else:
        y_pred_cls = y_pred.astype(int)

    y_true_cls = y_true.astype(int)

    # ── Compute confusion matrix ──────────────────────────────────────────────
    n_classes = max(y_true_cls.max(), y_pred_cls.max()) + 1
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(n_classes)))

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Truncate class names if needed (for large multi-class)
    class_names = [str(c)[:20] for c in class_names[:n_classes]]
    if len(class_names) < n_classes:
        class_names += [str(i) for i in range(len(class_names), n_classes)]

    # ── Normalise ─────────────────────────────────────────────────────────────
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid div/0
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm.astype(float)

    # ── Build annotation text matrix ─────────────────────────────────────────
    # Show both percentage and raw count
    annotations = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm[0])):
            pct   = cm_display[i, j]
            count = cm[i, j]
            if normalize:
                row.append(f"{pct:.1%}<br>({count})")
            else:
                row.append(str(count))
        annotations.append(row)

    # Custom colourscale: dark (wrong) → green (correct)
    colorscale = [
        [0.0, "#1A1A2E"],
        [0.3, "#1E3A5F"],
        [0.6, "#2D6A4F"],
        [1.0, "#52B788"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=12 if n_classes <= 10 else 9),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="Fraction" if normalize else "Count",
                      tickfont=dict(color=COLORS["text"])),
        hoverongaps=False,
        hovertemplate=(
            "Actual: %{y}<br>Predicted: %{x}<br>"
            "Value: %{z:.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        xaxis=dict(title="Predicted Label", side="bottom",
                   tickfont=dict(size=11 if n_classes <= 10 else 8)),
        yaxis=dict(title="True Label",
                   tickfont=dict(size=11 if n_classes <= 10 else 8),
                   autorange="reversed"),
        height=max(350, min(60 * n_classes + 100, 600)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=60, l=80, r=20),
    )

    return fig
