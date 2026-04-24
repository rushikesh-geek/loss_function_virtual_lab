# =============================================================================
# plots/gradient_viz.py
# Gradient magnitude per layer per epoch visualisation.
# Shows vanishing / exploding gradient patterns across training.
# =============================================================================

import plotly.graph_objects as go
import numpy as np
from config import COLORS, PLOTLY_TEMPLATE, CHART_HEIGHT, PLOTLY_COLORS


def plot_gradient_magnitudes(
    gradient_history: dict,
    title: str = "Gradient Magnitude per Layer per Epoch",
    log_scale: bool = True,
) -> go.Figure:
    """
    Plot L2 gradient norm per trainable layer across training epochs.

    Each layer = one line. Log scale is usually needed because gradients
    span several orders of magnitude (demonstrates vanishing gradient).

    Parameters
    ----------
    gradient_history : dict[layer_name: str] → list[float] (per epoch)
    title            : chart title
    log_scale        : use log-y axis (recommended for multi-layer nets)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    if not gradient_history:
        fig.add_annotation(
            text="No gradient data — train the model first.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=COLORS["subtext"], size=14),
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=CHART_HEIGHT,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    for idx, (layer_name, norms) in enumerate(gradient_history.items()):
        epochs = list(range(1, len(norms) + 1))
        color  = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]

        # Shorten layer name for legend readability
        short_name = layer_name.replace("_", " ").title()[:20]

        fig.add_trace(go.Scatter(
            x=epochs,
            y=norms,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=short_name,
            hovertemplate=(
                f"Layer: {short_name}<br>"
                "Epoch: %{x}<br>"
                "‖∇‖: %{y:.5f}<extra></extra>"
            ),
        ))

    # ── Add annotation for vanishing gradient region ──────────────────────────
    all_vals = [v for norms in gradient_history.values() for v in norms]
    if all_vals and min(all_vals) < 1e-4 and max(all_vals) > 1e-4:
        fig.add_hline(
            y=1e-4,
            line_dash="dot",
            line_color=COLORS["wrong"],
            annotation_text="Vanishing gradient zone (< 1e-4)",
            annotation_font_color=COLORS["wrong"],
            annotation_position="top left",
        )

    yaxis_type = "log" if log_scale else "linear"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        xaxis=dict(title="Epoch", gridcolor="#2D3250"),
        yaxis=dict(title="‖Gradient‖ (L2 norm)", type=yaxis_type, gridcolor="#2D3250"),
        height=CHART_HEIGHT,
        legend=dict(
            orientation="v",
            font=dict(size=10),
            title=dict(text="Layers", font=dict(size=11)),
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig
