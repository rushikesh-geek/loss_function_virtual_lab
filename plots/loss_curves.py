# =============================================================================
# plots/loss_curves.py
# Animated Plotly loss & accuracy curves for the Training Dashboard tab.
# Supports single-loss and side-by-side comparison mode.
# =============================================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from config import COLORS, PLOTLY_TEMPLATE, CHART_HEIGHT


def plot_loss_curves(
    history: dict,
    title: str = "Training History",
    task_type: str = "classification",   # "classification" | "regression"
    animate: bool = True,
) -> go.Figure:
    """
    Plot training + validation loss (and optionally accuracy/MAE) curves.

    Parameters
    ----------
    history   : dict from trainer — keys like 'loss', 'val_loss', 'accuracy', etc.
    title     : chart title
    task_type : determines secondary metric label
    animate   : add Plotly animation frames (epoch-by-epoch reveal)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    epochs = list(range(1, len(history["loss"]) + 1))

    # Determine secondary metric
    if "accuracy" in history:
        secondary_key     = "accuracy"
        val_secondary_key = "val_accuracy"
        secondary_label   = "Accuracy"
    elif "mae" in history:
        secondary_key     = "mae"
        val_secondary_key = "val_mae"
        secondary_label   = "MAE"
    else:
        secondary_key = val_secondary_key = secondary_label = None

    has_secondary = secondary_key is not None and secondary_key in history

    # ── Create subplots ───────────────────────────────────────────────────────
    n_rows = 2 if has_secondary else 1
    subplot_titles = ["Loss", secondary_label] if has_secondary else ["Loss"]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
    )

    # ── Loss traces ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color=COLORS["loss_accent"], width=2.5),
            marker=dict(size=5),
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color=COLORS["val_accent"], width=2.5, dash="dot"),
                marker=dict(size=5),
                hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Secondary metric traces ────────────────────────────────────────────────
    if has_secondary:
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history[secondary_key],
                mode="lines+markers",
                name=f"Train {secondary_label}",
                line=dict(color=COLORS["accuracy"], width=2.5),
                marker=dict(size=5),
                hovertemplate=f"Epoch %{{x}}<br>{secondary_label}: %{{y:.4f}}<extra></extra>",
            ),
            row=2, col=1,
        )

        if val_secondary_key in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history[val_secondary_key],
                    mode="lines+markers",
                    name=f"Val {secondary_label}",
                    line=dict(color=COLORS["secondary"], width=2.5, dash="dot"),
                    marker=dict(size=5),
                    hovertemplate=f"Epoch %{{x}}<br>Val {secondary_label}: %{{y:.4f}}<extra></extra>",
                ),
                row=2, col=1,
            )

    # ── Annotation: best epoch ────────────────────────────────────────────────
    if "val_loss" in history and len(history["val_loss"]) > 0:
        best_epoch = int(np.argmin(history["val_loss"]))
        best_val   = history["val_loss"][best_epoch]
        fig.add_vline(
            x=best_epoch + 1,
            line_dash="dash",
            line_color=COLORS["secondary"],
            annotation_text=f"Best epoch ({best_epoch+1})",
            annotation_position="top right",
            row=1, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT * n_rows,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=50, r=20),
    )
    fig.update_xaxes(
        title_text="Epoch",
        row=n_rows, col=1,
        gridcolor="#2D3250",
        linecolor="#2D3250",
    )
    fig.update_yaxes(gridcolor="#2D3250", linecolor="#2D3250")

    return fig


def plot_comparison_curves(
    history_a: dict,
    history_b: dict,
    label_a: str = "Loss A",
    label_b: str = "Loss B",
    task_type: str = "classification",
) -> go.Figure:
    """
    Side-by-side comparison of two training histories.
    Left: Loss curves. Right: Accuracy/MAE curves.
    """
    has_acc = "accuracy" in history_a
    metric   = "accuracy" if has_acc else "mae"
    m_label  = "Accuracy" if has_acc else "MAE"

    epochs_a = list(range(1, len(history_a["loss"]) + 1))
    epochs_b = list(range(1, len(history_b["loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Validation Loss", f"Validation {m_label}"],
        horizontal_spacing=0.1,
    )

    colors_a = COLORS["loss_accent"]
    colors_b = COLORS["primary"]

    # ── Loss ─────────────────────────────────────────────────────────────────
    for hist, epochs, color, label in [
        (history_a, epochs_a, colors_a, label_a),
        (history_b, epochs_b, colors_b, label_b),
    ]:
        val_key = "val_loss"
        if val_key in hist:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=hist[val_key], name=f"{label} (val)",
                    line=dict(color=color, width=2.5),
                    mode="lines+markers",
                    hovertemplate=f"{label}<br>Epoch %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ── Metric ───────────────────────────────────────────────────────────────
    val_metric = f"val_{metric}"
    for hist, epochs, color, label in [
        (history_a, epochs_a, colors_a, label_a),
        (history_b, epochs_b, colors_b, label_b),
    ]:
        if val_metric in hist:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=hist[val_metric], name=f"{label} (val {m_label})",
                    line=dict(color=color, width=2.5),
                    mode="lines+markers",
                    hovertemplate=f"{label}<br>Epoch %{{x}}<br>{m_label}: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=2,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Comparison: Training Histories", font=dict(size=16, color=COLORS["text"])),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=50, r=20),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="#2D3250", linecolor="#2D3250")
    fig.update_yaxes(gridcolor="#2D3250", linecolor="#2D3250")

    return fig
