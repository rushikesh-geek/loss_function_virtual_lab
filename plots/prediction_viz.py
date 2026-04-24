# =============================================================================
# plots/prediction_viz.py
# Prediction visualisations for all experiment types:
#   • MNIST image grid (5×5, green/red borders)
#   • Binary probability bar chart with threshold line
#   • Regression scatter + residual histogram
#   • Autoencoder original vs reconstructed image grid
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS, PLOTLY_TEMPLATE, CHART_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# MNIST Image Grid (5×5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mnist_grid(
    X_images: np.ndarray,    # (N, 28, 28, 1) or (N, 784)
    y_true: np.ndarray,      # integer labels
    y_pred_probs: np.ndarray,# (N, 10) softmax probabilities
    n_show: int = 25,
    title: str = "Prediction Grid — Green=Correct, Red=Wrong",
) -> go.Figure:
    """
    Show a 5×5 grid of MNIST images with colour-coded prediction borders.
    Uses Plotly image traces for interactivity.
    """
    import math

    n_cols = 5
    n_rows = math.ceil(n_show / n_cols)

    # Reshape images if flat
    if X_images.ndim == 2:
        imgs = X_images[:n_show].reshape(-1, 28, 28)
    else:
        imgs = X_images[:n_show, ..., 0]   # drop channel dim

    y_pred_cls = np.argmax(y_pred_probs[:n_show], axis=1)
    y_true_cls = y_true[:n_show].astype(int)

    # ── Build subplot grid ────────────────────────────────────────────────────
    titles = [
        f"T:{y_true_cls[i]} P:{y_pred_cls[i]}"
        for i in range(min(n_show, len(X_images)))
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    for idx in range(min(n_show, len(X_images))):
        r = idx // n_cols + 1
        c = idx % n_cols  + 1
        correct = (y_pred_cls[idx] == y_true_cls[idx])

        # Colour-coded: correct = greenish, wrong = reddish
        # We apply colour tint by manipulating the image RGB channels
        raw = imgs[idx]   # (28, 28) float [0,1]
        img_rgb = np.stack([raw, raw, raw], axis=-1)  # (28, 28, 3)

        if correct:
            img_rgb[:, :, 0] = raw * 0.4    # reduce red
            img_rgb[:, :, 1] = np.clip(raw + 0.2, 0, 1)   # boost green
            img_rgb[:, :, 2] = raw * 0.4
        else:
            img_rgb[:, :, 0] = np.clip(raw + 0.2, 0, 1)   # boost red
            img_rgb[:, :, 1] = raw * 0.4
            img_rgb[:, :, 2] = raw * 0.4

        img_uint8 = (img_rgb * 255).astype(np.uint8)

        fig.add_trace(
            go.Image(
                z=img_uint8,
                hovertemplate=(
                    f"True: {y_true_cls[idx]}<br>"
                    f"Pred: {y_pred_cls[idx]}<br>"
                    f"Conf: {y_pred_probs[idx, y_pred_cls[idx]]:.1%}<br>"
                    f"{'✅ Correct' if correct else '❌ Wrong'}<extra></extra>"
                ),
            ),
            row=r, col=c,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        height=160 * n_rows + 60,
        margin=dict(t=60, b=20, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Binary Probability Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_binary_probabilities(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,   # (N,) probabilities
    threshold: float = 0.5,
    n_show: int = 50,
    title: str = "Predicted Probabilities vs Decision Threshold",
) -> go.Figure:
    """
    Bar chart of predicted probabilities for N samples.
    Vertical threshold line shows current decision boundary.
    Bars are coloured: green=correct, red=wrong.
    """
    n  = min(n_show, len(y_true))
    idxs = np.arange(n)

    probs     = y_pred_probs[:n].ravel()
    labels    = y_true[:n].astype(int)
    predicted = (probs >= threshold).astype(int)
    correct   = (predicted == labels)

    bar_colors = [
        COLORS["correct"] if c else COLORS["wrong"] for c in correct
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=idxs,
        y=probs,
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate=(
            "Sample %{x}<br>"
            "P(positive): %{y:.3f}<br>"
            "True label: %{customdata[0]}<br>"
            "Prediction: %{customdata[1]}<extra></extra>"
        ),
        customdata=np.stack([labels, predicted], axis=1),
        name="Predicted Probability",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLORS["val_accent"],
        line_width=2,
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_font_color=COLORS["val_accent"],
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        xaxis_title="Sample Index",
        yaxis_title="P(Positive Class)",
        yaxis=dict(range=[0, 1.05]),
        height=CHART_HEIGHT,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=50, r=20),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Regression Scatter + Residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    loss_name: str = "",
) -> go.Figure:
    """
    Scatter of predicted vs actual values plus residual histogram.
    Perfect prediction = identity line (y=x).
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    residuals = y_pred - y_true

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Predicted vs Actual", "Residual Distribution"],
        horizontal_spacing=0.12,
    )

    # ── Scatter ───────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred,
            mode="markers",
            marker=dict(
                color=np.abs(residuals),
                colorscale="RdYlGn_r",
                showscale=True,
                size=5,
                opacity=0.7,
                colorbar=dict(title="|Residual|", x=0.46),
            ),
            hovertemplate="True: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>",
            name="Predictions",
        ),
        row=1, col=1,
    )

    # Identity line
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    fig.add_trace(
        go.Scatter(
            x=lim, y=lim,
            mode="lines",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            name="Perfect Prediction",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Residual histogram ────────────────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=40,
            marker_color=COLORS["primary"],
            opacity=0.8,
            name="Residuals",
            hovertemplate="Residual: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=2,
    )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["val_accent"], row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"Regression Predictions — {loss_name}" if loss_name else title,
            font=dict(size=14, color=COLORS["text"]),
        ),
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=50, r=20),
    )
    fig.update_xaxes(gridcolor="#2D3250")
    fig.update_yaxes(gridcolor="#2D3250")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder Original vs Reconstructed
# ─────────────────────────────────────────────────────────────────────────────

def plot_autoencoder_reconstructions(
    X_original: np.ndarray,      # (N, 28, 28, 1)
    X_reconstructed: np.ndarray, # (N, 28, 28, 1)
    n_pairs: int = 8,
    title: str = "Original vs Reconstructed",
) -> go.Figure:
    """
    Show N pairs of (original, reconstructed) MNIST images.
    Top row: originals. Bottom row: reconstructions.
    """
    n = min(n_pairs, len(X_original))

    fig = make_subplots(
        rows=2, cols=n,
        subplot_titles=(
            [f"Orig {i+1}" for i in range(n)] +
            [f"Recon {i+1}" for i in range(n)]
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for col_idx in range(n):
        for row_idx, (arr, label) in enumerate([
            (X_original,      "Original"),
            (X_reconstructed, "Reconstructed"),
        ]):
            img = arr[col_idx]
            if img.ndim == 3:
                img = img[..., 0]  # drop channel

            img_rgb   = np.stack([img, img, img], axis=-1)
            img_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

            mse_val = float(np.mean((X_original[col_idx] - X_reconstructed[col_idx])**2))

            fig.add_trace(
                go.Image(
                    z=img_uint8,
                    hovertemplate=(
                        f"{label}<br>"
                        f"Pixel MSE: {mse_val:.5f}<extra></extra>"
                    ),
                ),
                row=row_idx + 1, col=col_idx + 1,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        height=330,
        margin=dict(t=60, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


def plot_predictions(experiment: int, **kwargs):
    """Dispatcher: route to the correct plot function based on experiment number."""
    if experiment in (1,):
        return plot_mnist_grid(**kwargs)
    elif experiment == 2:
        return plot_binary_probabilities(**kwargs)
    elif experiment == 3:
        return plot_regression_predictions(**kwargs)
    elif experiment == 5:
        return plot_autoencoder_reconstructions(**kwargs)
    else:
        # Default fallback
        return go.Figure()
