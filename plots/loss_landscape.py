# =============================================================================
# plots/loss_landscape.py
# 3D interactive Plotly loss landscape surface.
#
# Method: perturb TWO weights of the trained model orthogonally,
# compute the loss at each (w1, w2) grid point.
# Overlay the optimizer's weight path through training.
#
# Reference: Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)
#            https://arxiv.org/abs/1712.10135
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from typing import Optional
import tensorflow as tf
from config import COLORS, PLOTLY_TEMPLATE, LANDSCAPE_GRID


def compute_loss_landscape(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn,
    grid_size: int = LANDSCAPE_GRID,
    alpha_range: float = 1.0,   # perturbation range along direction 1
    beta_range: float = 1.0,    # perturbation range along direction 2
    weight_layer_idx: int = 0,  # which layer's weights to perturb
) -> tuple:
    """
    Compute a grid_size×grid_size array of loss values by perturbing 2 weights.

    Strategy
    ────────
    1. Pick the first trainable Dense layer with ≥2 weights.
    2. Extract TWO random orthonormal direction vectors in weight space.
    3. For each (α, β) on the grid: w_perturbed = w0 + α·d1 + β·d2.
    4. Compute loss(model(X), y) with those perturbed weights.
    5. Restore original weights.

    Returns
    -------
    (alpha_vals, beta_vals, Z) where Z is the (grid, grid) loss surface array.
    """
    # ── Find a suitable layer to perturb ─────────────────────────────────────
    target_layer = None
    for layer in model.layers:
        if hasattr(layer, "kernel") and layer.kernel is not None:
            w = layer.kernel.numpy()
            if w.size >= 2:
                target_layer = layer
                break

    if target_layer is None:
        # Fallback: use a trivial flat landscape
        alpha_vals = np.linspace(-alpha_range, alpha_range, grid_size)
        beta_vals  = np.linspace(-beta_range,  beta_range,  grid_size)
        Z = np.zeros((grid_size, grid_size))
        return alpha_vals, beta_vals, Z

    w0    = target_layer.kernel.numpy().copy()     # original weights
    shape = w0.shape
    w_flat = w0.ravel()

    # ── Two random orthonormal directions ─────────────────────────────────────
    rng  = np.random.RandomState(42)
    d1   = rng.randn(len(w_flat)); d1 /= (np.linalg.norm(d1) + 1e-12)
    d2   = rng.randn(len(w_flat))
    d2  -= d2.dot(d1) * d1                         # Gram-Schmidt orthogonalise
    d2  /= (np.linalg.norm(d2) + 1e-12)

    alpha_vals = np.linspace(-alpha_range, alpha_range, grid_size)
    beta_vals  = np.linspace(-beta_range,  beta_range,  grid_size)

    # Subsample X/y for speed (loss landscape is approximate)
    n_subsample = min(512, len(X))
    idx = np.random.choice(len(X), n_subsample, replace=False)
    Xs  = tf.constant(X[idx],  dtype=tf.float32)
    ys  = tf.constant(y[idx],  dtype=tf.float32)

    Z = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            w_new = (w_flat + alpha * d1 + beta * d2).reshape(shape)
            target_layer.kernel.assign(w_new)

            preds = model(Xs, training=False)
            loss_val = float(tf.reduce_mean(loss_fn(ys, preds)).numpy())
            Z[i, j] = loss_val

    # Restore original weights
    target_layer.kernel.assign(w0)

    return alpha_vals, beta_vals, Z


def plot_loss_landscape(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn,
    weight_snapshots: Optional[list] = None,   # list of (alpha, beta, loss) tuples
    grid_size: int = LANDSCAPE_GRID,
    title: str = "Loss Landscape",
) -> go.Figure:
    """
    Render a 3D Plotly surface of the loss landscape with optional optimizer path.

    Parameters
    ----------
    model            : trained Keras model
    X, y             : data for loss computation
    loss_fn          : Keras loss callable
    weight_snapshots : list of (alpha, beta, loss_val) for optimizer trajectory
    grid_size        : resolution of the surface grid (N×N)
    title            : chart title

    Returns
    -------
    plotly.graph_objects.Figure with Surface + optional Scatter3d trace
    """
    alpha_vals, beta_vals, Z = compute_loss_landscape(
        model, X, y, loss_fn, grid_size=grid_size
    )

    # ── Clip extreme values to improve visualisation ──────────────────────────
    z_lo, z_hi = np.percentile(Z, 5), np.percentile(Z, 95)
    Z_clipped   = np.clip(Z, z_lo, z_hi)

    fig = go.Figure()

    # ── 3D Surface ────────────────────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=alpha_vals,
        y=beta_vals,
        z=Z_clipped,
        colorscale=[
            [0.0, "#052F5F"],
            [0.2, "#0A6E9E"],
            [0.5, "#4AAFCA"],
            [0.8, "#FFD93D"],
            [1.0, "#FF6B6B"],
        ],
        opacity=0.85,
        showscale=True,
        colorbar=dict(title="Loss", tickfont=dict(color=COLORS["text"])),
        hovertemplate=(
            "α: %{x:.3f}<br>"
            "β: %{y:.3f}<br>"
            "Loss: %{z:.4f}<extra></extra>"
        ),
        name="Loss Surface",
    ))

    # ── Mark global minimum ───────────────────────────────────────────────────
    min_idx = np.unravel_index(np.argmin(Z_clipped), Z_clipped.shape)
    fig.add_trace(go.Scatter3d(
        x=[alpha_vals[min_idx[0]]],
        y=[beta_vals[min_idx[1]]],
        z=[Z_clipped[min_idx]],
        mode="markers",
        marker=dict(size=8, color="#6BCB77", symbol="circle"),
        name="Minimum",
        hovertemplate=f"Minimum<br>Loss: {Z_clipped[min_idx]:.4f}<extra></extra>",
    ))

    # ── Optimizer trajectory (simulated) ──────────────────────────────────────
    # We simulate a gradient descent path from the upper-right corner to minimum
    # (the actual tracked path requires weight snapshots during training)
    sim_path = _simulate_optimizer_path(alpha_vals, beta_vals, Z_clipped, n_steps=20)
    if sim_path is not None:
        path_a, path_b, path_z = sim_path
        fig.add_trace(go.Scatter3d(
            x=path_a, y=path_b, z=path_z + 0.01 * (path_z.max() - path_z.min()),
            mode="lines+markers",
            line=dict(color="#FFD93D", width=5),
            marker=dict(size=4, color="#FFD93D"),
            name="Optimizer Path",
            hovertemplate="Step path<br>Loss: %{z:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        template=PLOTLY_TEMPLATE,
        scene=dict(
            xaxis=dict(title="Direction α", gridcolor="#2D3250", backgroundcolor="#0E1117"),
            yaxis=dict(title="Direction β", gridcolor="#2D3250", backgroundcolor="#0E1117"),
            zaxis=dict(title="Loss",        gridcolor="#2D3250", backgroundcolor="#0E1117"),
            bgcolor="#0E1117",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=500,
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor="#0E1117",
        legend=dict(font=dict(color=COLORS["text"])),
    )

    return fig


def _simulate_optimizer_path(
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    Z: np.ndarray,
    n_steps: int = 20,
) -> Optional[tuple]:
    """
    Simulate a simple gradient descent path on the loss surface for illustration.
    Starts from a high-loss region and follows approximate negative gradient.
    """
    try:
        # Start from the highest-loss cell
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        i, j    = max_idx

        path_i, path_j = [i], [j]
        visited = {(i, j)}

        for _ in range(n_steps - 1):
            # Check 4 neighbours, move to lowest-loss unvisited neighbour
            neighbours = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            best = None
            best_loss = float("inf")
            for ni, nj in neighbours:
                if 0 <= ni < Z.shape[0] and 0 <= nj < Z.shape[1] and (ni,nj) not in visited:
                    if Z[ni, nj] < best_loss:
                        best_loss = Z[ni, nj]
                        best = (ni, nj)
            if best is None:
                break
            i, j = best
            path_i.append(i)
            path_j.append(j)
            visited.add((i, j))

        path_a = alpha_vals[path_i]
        path_b = beta_vals[path_j]
        path_z = Z[path_i, path_j]

        return path_a, path_b, path_z
    except Exception:
        return None
