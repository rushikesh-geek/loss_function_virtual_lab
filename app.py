# =============================================================================
# app.py — Interactive Loss Function Virtual Lab
# Main Streamlit entry point.
#
# Streamlit run: streamlit run app.py
#
# Architecture overview:
#   sidebar  → experiment + hyperparameter selection
#   main     → 6 tabs: Dashboard, Predictions, Loss Analysis,
#              Loss Landscape, Compare Mode, Explainer
#
# All training results live in st.session_state so switching tabs
# does not trigger re-training.
# =============================================================================

import sys
import os

# ── Ensure all modules are importable from the virtual_lab/ root ─────────────
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Internal modules ──────────────────────────────────────────────────────────
from config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION,
    EXPERIMENTS, LOSS_METADATA, DEFAULTS,
    COLORS, CUSTOM_CSS,
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Loss Function Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═════════════════════════════════════════════════════════════════════════════
def init_session_state():
    """
    Initialise all session_state keys with defaults.
    Called once at startup; subsequent calls are no-ops (keys already set).
    """
    defaults = {
        # Experiment selection
        "experiment_id":    1,
        "model_type":       "MLP",
        "loss_name":        "Cross Entropy",
        "loss_name_b":      "MSE",          # second loss for comparison mode
        "comparison_mode":  False,
        "optimizer":        DEFAULTS["optimizer"],
        "epochs":           DEFAULTS["epochs"],
        "batch_size":       DEFAULTS["batch_size"],
        "learning_rate":    DEFAULTS["learning_rate"],
        "dropout":          DEFAULTS["dropout"],
        "huber_delta":      DEFAULTS["huber_delta"],
        "focal_alpha":      DEFAULTS["focal_alpha"],
        "focal_gamma":      DEFAULTS["focal_gamma"],
        "threshold":        DEFAULTS["threshold"],
        "n_outliers":       DEFAULTS["n_outliers"],
        "outlier_magnitude":DEFAULTS["outlier_magnitude"],
        "fraud_ratio":      DEFAULTS["fraud_ratio"],
        "show_dataset":     False,

        # Training results (populated after [TRAIN] click)
        "trained":          False,
        "history":          None,
        "history_b":        None,            # comparison mode history
        "gradient_history": None,
        "y_pred":           None,
        "y_pred_b":         None,
        "y_true_test":      None,
        "dataset":          None,
        "metrics":          None,
        "metrics_b":        None,
        "model":            None,
        "model_b":          None,

        # Autoencoder-specific
        "encoder":          None,
        "latent_codes":     None,
        "latent_labels":    None,
        "reconstructed":    None,

        # Experiment 4 — imbalanced classification extras
        # Stored after training so the threshold slider re-uses the auto-detected
        # value on tab switches without re-running threshold computation.
        "exp4_best_threshold":   0.20,   # PR-curve F1-optimal threshold (model A)
        "exp4_best_threshold_b": 0.20,   # same for comparison model B

        # Experiment 5 — autoencoder extras
        "latent_dim":   DEFAULTS["latent_dim"],
        "vae_beta":     DEFAULTS["vae_beta"],
        "noise_factor": DEFAULTS["noise_factor"],
        "exp5_mode":    "MSE Reconstruction",   # tracks the active AE loss mode
        "vae_kl_history":    None,              # for split loss curve (VAE)
        "vae_recon_history": None,              # reconstruction only curve
        "X_train_noisy":     None,              # noisy inputs for Denoising AE viz
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    """Render all sidebar controls and return the resulting parameter dict."""
    with st.sidebar:
        # App header
        st.markdown(f"""
        <div style="text-align:center; padding:12px 0 20px 0;">
            <div style="font-size:2.2rem;">🧪</div>
            <div style="font-size:1.1rem; font-weight:700; color:#7EC8E3;">
                Loss Function Lab
            </div>
            <div style="font-size:0.72rem; color:#A0A8C0;">
                v{APP_VERSION} · Deep Learning Teaching Tool
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Experiment selector ───────────────────────────────────────────────
        st.markdown("### 🔬 Experiment")
        exp_options = {
            f"Exp {k}: {v['icon']} {v['name']}": k
            for k, v in EXPERIMENTS.items()
        }
        selected_exp_label = st.selectbox(
            "Select Experiment",
            options=list(exp_options.keys()),
            index=st.session_state.experiment_id - 1,
            key="exp_selector",
            label_visibility="collapsed",
        )
        exp_id = exp_options[selected_exp_label]

        # Reset training state if experiment changed
        if exp_id != st.session_state.experiment_id:
            st.session_state.experiment_id  = exp_id
            st.session_state.trained        = False
            st.session_state.history        = None
            st.session_state.y_pred         = None
            _set_experiment_defaults(exp_id)

        exp_info = EXPERIMENTS[exp_id]

        st.markdown(
            f'<div class="exp-badge">{exp_info["icon"]} <b>{exp_info["name"]}</b>'
            f'<br><span style="font-size:0.8rem;color:#A0A8C0;">'
            f'{exp_info["dataset"]}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Dataset preview toggle ────────────────────────────────────────────
        st.session_state.show_dataset = st.toggle(
            "📋 Show Dataset Preview",
            value=st.session_state.show_dataset,
        )

        st.divider()

        # ── Model & Loss selectors ────────────────────────────────────────────
        st.markdown("### 🏗️ Model & Loss")

        model_type = st.selectbox(
            "Model Architecture",
            options=exp_info["models"],
            index=0,
        )
        st.session_state.model_type = model_type

        loss_name = st.selectbox(
            "Loss Function",
            options=exp_info["losses"],
            index=0,
        )
        st.session_state.loss_name = loss_name

        # Comparison mode toggle
        if len(exp_info["losses"]) >= 2:
            comp_mode = st.toggle(
                "⚖️ Comparison Mode (2 losses)",
                value=st.session_state.comparison_mode,
            )
            st.session_state.comparison_mode = comp_mode

            if comp_mode:
                loss_b_opts = [l for l in exp_info["losses"] if l != loss_name]
                if loss_b_opts:
                    loss_name_b = st.selectbox(
                        "Loss B (to compare)",
                        options=loss_b_opts,
                        index=0,
                    )
                    st.session_state.loss_name_b = loss_name_b

        st.divider()

        # ── Hyperparameters ───────────────────────────────────────────────────
        st.markdown("### ⚙️ Hyperparameters")

        optimizer = st.selectbox(
            "Optimizer",
            options=["Adam", "SGD", "RMSprop"],
            index=["Adam", "SGD", "RMSprop"].index(st.session_state.optimizer),
        )
        st.session_state.optimizer = optimizer

        epochs = st.slider(
            "Epochs",
            min_value=1, max_value=50,
            value=st.session_state.epochs, step=1,
        )
        st.session_state.epochs = epochs

        batch_size = st.selectbox(
            "Batch Size",
            options=[32, 64, 128, 256],
            index=[32, 64, 128, 256].index(st.session_state.batch_size),
        )
        st.session_state.batch_size = batch_size

        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            value=st.session_state.learning_rate,
            format_func=lambda v: f"{v:.0e}",
        )
        st.session_state.learning_rate = learning_rate

        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0, max_value=0.5,
            value=st.session_state.dropout, step=0.05,
        )
        st.session_state.dropout = dropout

        # ── Experiment-specific controls ──────────────────────────────────────
        if exp_id == 3:
            st.divider()
            st.markdown("### 🚀 Outlier Controls")
            n_out = st.slider(
                "Number of Outliers",
                min_value=0, max_value=200,
                value=st.session_state.n_outliers, step=5,
            )
            st.session_state.n_outliers = n_out

            out_mag = st.slider(
                "Outlier Magnitude (× max)",
                min_value=1.0, max_value=20.0,
                value=st.session_state.outlier_magnitude, step=0.5,
            )
            st.session_state.outlier_magnitude = out_mag

            huber_d = st.slider(
                "Huber δ (delta)",
                min_value=0.1, max_value=5.0,
                value=st.session_state.huber_delta, step=0.1,
            )
            st.session_state.huber_delta = huber_d

        elif exp_id == 4:
            st.divider()
            st.markdown("### ⚖️ Imbalance Controls")
            fraud_r = st.slider(
                "Fraud Ratio (%)",
                min_value=1, max_value=15, value=int(st.session_state.fraud_ratio * 100), step=1,
                help="Percentage of minority (fraud) samples in the dataset.",
            )
            st.session_state.fraud_ratio = fraud_r / 100.0

            if "Focal Loss" in exp_info["losses"]:
                st.markdown("**Focal Loss Parameters**")
                focal_a = st.slider(
                    "α (alpha) — class weight",
                    min_value=0.1, max_value=0.9,
                    value=st.session_state.focal_alpha, step=0.05,
                )
                st.session_state.focal_alpha = focal_a

                focal_g = st.slider(
                    "γ (gamma) — focus strength",
                    min_value=0.5, max_value=5.0,
                    value=st.session_state.focal_gamma, step=0.25,
                )
                st.session_state.focal_gamma = focal_g

        elif exp_id == 5:
            st.divider()
            st.markdown("### 🔬 Autoencoder Controls")

            latent_dim = st.slider(
                "Latent Dimension",
                min_value=2, max_value=64,
                value=st.session_state.latent_dim, step=2,
                help=(
                    "The bottleneck size — how many numbers encode each image.\n"
                    "Set to **2** for a 2D latent scatter plot.\n"
                    "Higher dims → better quality, less visualisable."
                ),
            )
            st.session_state.latent_dim = latent_dim

            # VAE-specific: beta slider (shown only when VAE is selected)
            if "VAE" in st.session_state.loss_name:
                vae_beta = st.slider(
                    "β (Beta) — KL weight",
                    min_value=0.1, max_value=5.0,
                    value=st.session_state.vae_beta, step=0.1,
                    help=(
                        "β=1 → original VAE.  β>1 → β-VAE: stronger KL regularisation "
                        "→ more disentangled latent space. β<1 → sharper reconstructions."
                    ),
                )
                st.session_state.vae_beta = vae_beta

            # Denoising AE: noise factor slider (shown only for Denoising mode)
            if "Denoising" in st.session_state.loss_name:
                noise_f = st.slider(
                    "Noise Factor (σ)",
                    min_value=0.1, max_value=0.6,
                    value=st.session_state.noise_factor, step=0.05,
                    help=(
                        "Standard deviation of Gaussian noise added to input.\n"
                        "0.1 = mild, 0.5 = heavy. Higher → harder task → more robust features."
                    ),
                )
                st.session_state.noise_factor = noise_f

        st.divider()

        # ── TRAIN button ─────────────────────────────────────────────────────
        train_clicked = st.button(
            "🚀 TRAIN MODEL",
            use_container_width=True,
            type="primary",
        )

        # Status area (shown during training)
        train_status_placeholder = st.empty()

        # Threshold slider (Experiment 2 — no re-train needed)
        if exp_id == 2 and st.session_state.trained:
            st.divider()
            st.markdown("### 🎯 Decision Threshold")
            threshold = st.slider(
                "Threshold (real-time)",
                min_value=0.05, max_value=0.95,
                value=st.session_state.threshold, step=0.05,
            )
            st.session_state.threshold = threshold

        # ROOT CAUSE 3: Experiment 4 threshold slider — range 0.05→0.50
        # The default 0.50 is badly calibrated for imbalanced data.
        # After training, exp4_best_threshold is set from the PR-curve F1
        # optimum.  Here we allow the student to explore nearby values
        # interactively (no re-train needed — only the decision boundary changes).
        if exp_id == 4 and st.session_state.trained:
            st.divider()
            st.markdown("### 🎯 Fraud Detection Threshold")
            best_thr = st.session_state.exp4_best_threshold
            st.info(
                f"🤖 Auto-detected best threshold: **{best_thr:.3f}**  \n"
                "*(maximises F1 on the fraud class via PR curve)*",
                icon="🎯",
            )
            # Root Cause 3: Use 0.05–0.50 as the slider range.
            # 0.50 is the upper bound because higher thresholds on imbalanced data
            # almost always predict zero fraud (model collapse).
            exp4_threshold = st.slider(
                "Decision Threshold",
                min_value=0.05, max_value=0.50,
                value=float(np.clip(best_thr, 0.05, 0.50)),
                step=0.01,
                key="exp4_threshold_sidebar",
                help=(
                    "Lower threshold → catch more fraud (higher recall, lower precision).  \n"
                    "Higher threshold → fewer false alarms (higher precision, lower recall).  \n"
                    "The auto-detected value maximises F1 = harmonic mean of both."
                ),
            )
            st.session_state.threshold = exp4_threshold

    return train_clicked, train_status_placeholder


def _set_experiment_defaults(exp_id: int):
    """Switch to experiment-appropriate defaults for loss and model."""
    exp = EXPERIMENTS[exp_id]
    st.session_state.loss_name  = exp["losses"][0]
    st.session_state.model_type = exp["models"][0]
    st.session_state.comparison_mode = False


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════
def run_training(status_placeholder):
    """
    Dispatch to the correct training function based on experiment_id.
    Results stored in st.session_state.
    """
    exp_id = st.session_state.experiment_id

    progress_bar = st.progress(0)
    status_text  = status_placeholder.empty()

    def update_progress(frac):
        progress_bar.progress(min(frac, 1.0))

    def update_status(text):
        status_text.text(text)

    try:
        with st.spinner("🔧 Loading dataset & building model…"):
            if exp_id == 1:
                results = _train_exp1(update_progress, update_status)
            elif exp_id == 2:
                results = _train_exp2(update_progress, update_status)
            elif exp_id == 3:
                results = _train_exp3(update_progress, update_status)
            elif exp_id == 4:
                results = _train_exp4(update_progress, update_status)
            elif exp_id == 5:
                results = _train_exp5(update_progress, update_status)

        # Store all results
        for k, v in results.items():
            st.session_state[k] = v

        st.session_state.trained = True
        progress_bar.progress(1.0)
        status_text.text("✅ Training complete!")
        st.success("🎉 Training finished! Explore the tabs above.")

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)

    finally:
        import time; time.sleep(0.5)
        progress_bar.empty()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — MNIST multi-class
# ─────────────────────────────────────────────────────────────────────────────
def _train_exp1(prog_fn, status_fn) -> dict:
    from datasets.mnist_loader import load_mnist
    from models.mlp import build_mlp
    from models.cnn import build_cnn
    from training.trainer import Trainer
    from utils.metrics import compute_metrics
    from losses.loss_registry import get_loss

    ss = st.session_state

    data = load_mnist(flatten=(ss.model_type == "MLP"), subset=8000)

    if ss.model_type == "MLP":
        X_tr = data["X_train_flat"]
        X_te = data["X_test_flat"]
    else:
        X_tr = data["X_train"]
        X_te = data["X_test"]

    y_tr  = data["y_train_cat"]   # one-hot for CE
    y_te  = data["y_test_cat"]
    y_int = data["y_test"]        # integer labels for metrics

    def build(loss_nm):
        if ss.model_type == "MLP":
            return build_mlp(
                input_dim=784, output_dim=10,
                task_type="multiclass",
                hidden_units=(256, 128, 64),
                dropout_rate=ss.dropout,
                learning_rate=ss.learning_rate,
                optimizer_name=ss.optimizer,
                loss_name=loss_nm,
            )
        else:
            return build_cnn(
                input_shape=(28, 28, 1), n_classes=10,
                dropout_rate=ss.dropout,
                learning_rate=ss.learning_rate,
                optimizer_name=ss.optimizer,
                loss_name=loss_nm,
            )

    model_a = build(ss.loss_name)
    trainer = Trainer(model_a, X_tr, y_tr)
    hist_a, grad_hist = trainer.train(
        epochs=ss.epochs, batch_size=ss.batch_size,
        progress_fn=prog_fn, status_fn=status_fn,
    )
    y_pred_a = trainer.predict(X_te)
    metrics_a = compute_metrics(y_int, y_pred_a, task_type="multiclass")

    results = {
        "dataset": data,
        "history": hist_a,
        "gradient_history": grad_hist,
        "y_pred": y_pred_a,
        "y_true_test": y_int,
        "metrics": metrics_a,
        "model": model_a,
    }

    # Comparison mode
    if ss.comparison_mode:
        model_b = build(ss.loss_name_b)
        # For MSE in classification, use integer labels (no one-hot needed)
        y_tr_b = data["y_train"] if ss.loss_name_b == "MSE" else y_tr
        y_te_b = y_int            if ss.loss_name_b == "MSE" else y_te

        # MSE on multiclass: we use raw integer labels... but model still has softmax
        # Reshape y to match output shape
        if ss.loss_name_b == "MSE":
            from tensorflow import keras
            y_tr_b = keras.utils.to_categorical(data["y_train"], 10).astype("float32")
        trainer_b = Trainer(model_b, X_tr, y_tr_b)
        hist_b, _ = trainer_b.train(
            epochs=ss.epochs, batch_size=ss.batch_size,
            progress_fn=prog_fn, status_fn=status_fn,
            record_gradients=False,
        )
        y_pred_b  = trainer_b.predict(X_te)
        metrics_b = compute_metrics(y_int, y_pred_b, task_type="multiclass")
        results.update({
            "history_b": hist_b,
            "y_pred_b":  y_pred_b,
            "metrics_b": metrics_b,
            "model_b":   model_b,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — Breast Cancer binary
# ─────────────────────────────────────────────────────────────────────────────
def _train_exp2(prog_fn, status_fn) -> dict:
    from datasets.binary_loader import load_breast_cancer_data
    from models.mlp import build_mlp
    from training.trainer import Trainer
    from utils.metrics import compute_metrics

    ss   = st.session_state
    data = load_breast_cancer_data()

    model = build_mlp(
        input_dim=data["n_features"], output_dim=1,
        task_type="binary",
        hidden_units=(128, 64, 32),
        dropout_rate=ss.dropout,
        learning_rate=ss.learning_rate,
        optimizer_name=ss.optimizer,
        loss_name="Binary Cross Entropy",
    )

    trainer = Trainer(model, data["X_train"], data["y_train"])
    hist, grad_hist = trainer.train(
        epochs=ss.epochs, batch_size=ss.batch_size,
        progress_fn=prog_fn, status_fn=status_fn,
    )

    y_pred   = trainer.predict(data["X_test"])
    metrics  = compute_metrics(
        data["y_test"], y_pred,
        task_type="binary", threshold=ss.threshold,
    )

    return {
        "dataset":          data,
        "history":          hist,
        "gradient_history": grad_hist,
        "y_pred":           y_pred,
        "y_true_test":      data["y_test"],
        "metrics":          metrics,
        "model":            model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — California Housing regression
# ─────────────────────────────────────────────────────────────────────────────
def _train_exp3(prog_fn, status_fn) -> dict:
    from datasets.regression_loader import load_california_housing_data
    from models.mlp import build_mlp
    from training.trainer import Trainer
    from utils.metrics import compute_metrics

    ss   = st.session_state
    data = load_california_housing_data(
        n_outliers=ss.n_outliers,
        outlier_magnitude=ss.outlier_magnitude,
    )

    def build(loss_nm):
        return build_mlp(
            input_dim=data["n_features"], output_dim=1,
            task_type="regression",
            hidden_units=(128, 64, 32),
            dropout_rate=ss.dropout,
            learning_rate=ss.learning_rate,
            optimizer_name=ss.optimizer,
            loss_name=loss_nm,
        )

    model_a = build(ss.loss_name)
    trainer = Trainer(model_a, data["X_train"], data["y_train"])
    hist_a, grad_hist = trainer.train(
        epochs=ss.epochs, batch_size=ss.batch_size,
        progress_fn=prog_fn, status_fn=status_fn,
    )
    y_pred_a  = trainer.predict(data["X_test"])
    metrics_a = compute_metrics(data["y_test"], y_pred_a, task_type="regression")

    results = {
        "dataset":          data,
        "history":          hist_a,
        "gradient_history": grad_hist,
        "y_pred":           y_pred_a,
        "y_true_test":      data["y_test"],
        "metrics":          metrics_a,
        "model":            model_a,
    }

    # Comparison Mode — always compare MSE vs Huber for Exp 3
    if ss.comparison_mode:
        alt_loss = "Huber" if ss.loss_name == "MSE" else "MSE"
        model_b  = build(alt_loss)
        trainer_b = Trainer(model_b, data["X_train"], data["y_train"])
        hist_b, _ = trainer_b.train(
            epochs=ss.epochs, batch_size=ss.batch_size,
            progress_fn=prog_fn, status_fn=status_fn,
            record_gradients=False,
        )
        y_pred_b  = trainer_b.predict(data["X_test"])
        metrics_b = compute_metrics(data["y_test"], y_pred_b, task_type="regression")
        results.update({
            "history_b": hist_b,
            "y_pred_b":  y_pred_b,
            "metrics_b": metrics_b,
            "model_b":   model_b,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4 — Fraud Detection (Imbalanced Classification)
# Fixes applied here:
#   ROOT CAUSE 1 — class_weight passed to Trainer.train()
#   ROOT CAUSE 2 — focal_alpha derived from actual fraud_ratio
#   ROOT CAUSE 3 — best_threshold from PR-curve F1 maximisation
# ─────────────────────────────────────────────────────────────────────────────
def _train_exp4(prog_fn, status_fn) -> dict:
    from datasets.fraud_loader import load_fraud_data
    from models.mlp import build_mlp
    from training.trainer import Trainer
    from utils.metrics import compute_metrics, best_threshold_from_proba

    ss   = st.session_state
    data = load_fraud_data(fraud_ratio=ss.fraud_ratio)

    # ── ROOT CAUSE 1: retrieve pre-computed class weights from the dataset ────
    # class_weight_dict = {0: w_normal, 1: w_fraud}
    # w_fraud >> w_normal so both classes contribute equal total gradient.
    # This is the FIRST line of defence against model collapse.
    class_weight_dict = data["class_weight_dict"]

    # ── ROOT CAUSE 2: set focal alpha from the actual imbalance ratio ─────────
    # alpha should reflect how RARE the minority class is.
    # alpha = 1 - fraud_ratio  (e.g. 0.95 for 5% fraud)
    # This means: loss on a fraud sample ≈ 0.95,
    #             loss on a normal sample ≈ 0.05.
    # The user can still OVERRIDE via the sidebar slider (ss.focal_alpha),
    # but we set a sensible data-driven default when Focal Loss is selected.
    imbalance_alpha = float(np.clip(
        1.0 - data["fraud_ratio"],   # e.g. 1 - 0.05 = 0.95
        0.70,                         # floor: never go below 0.70
        0.99,                         # ceiling: leave some gradient for normal
    ))

    # Seed focal_alpha in session state to the imbalance-ratio value
    # so the sidebar slider starts at the correct position on first run.
    if abs(ss.focal_alpha - 0.25) < 0.01:   # still at old default → update it
        st.session_state.focal_alpha = imbalance_alpha

    def build(loss_nm):
        # For Focal Loss: use the imbalance-ratio alpha UNLESS the user has
        # explicitly changed the sidebar slider.
        if loss_nm == "Focal Loss":
            effective_alpha = st.session_state.focal_alpha
            effective_gamma = ss.focal_gamma
        else:
            # BCE also benefits from a tuned alpha, but we intentionally use
            # the old default here so students can SEE why BCE fails at 5% fraud.
            effective_alpha = 0.25
            effective_gamma = 2.0

        return build_mlp(
            input_dim      = data["n_features"],
            output_dim     = 1,
            task_type      = "binary",
            hidden_units   = (128, 64, 32),
            dropout_rate   = ss.dropout,
            learning_rate  = ss.learning_rate,
            optimizer_name = ss.optimizer,
            loss_name      = loss_nm,
            focal_alpha    = effective_alpha,
            focal_gamma    = effective_gamma,
        )

    # ── Train model A ─────────────────────────────────────────────────────────
    model_a = build(ss.loss_name)
    trainer = Trainer(model_a, data["X_train"], data["y_train"])
    hist_a, grad_hist = trainer.train(
        epochs       = ss.epochs,
        batch_size   = ss.batch_size,
        progress_fn  = prog_fn,
        status_fn    = status_fn,
        class_weight = class_weight_dict,   # ROOT CAUSE 1
    )
    y_pred_a = trainer.predict(data["X_test"])

    # ── ROOT CAUSE 3: compute best threshold from actual model predictions ────
    # Now that we have real model output, find the threshold that maximises F1
    # on the FRAUD class using the Precision-Recall curve.
    # This is more reliable than the pre-training LR probe in fraud_loader.py
    # because it uses the trained network's actual probability distribution.
    best_thr_a = best_threshold_from_proba(
        data["y_test"], y_pred_a.ravel(), pos_label=1
    )
    st.session_state.exp4_best_threshold = best_thr_a
    st.session_state.threshold           = best_thr_a   # initialise the slider

    metrics_a = compute_metrics(
        data["y_test"], y_pred_a,
        task_type = "binary",
        threshold = best_thr_a,    # use optimal, not 0.50
    )

    results = {
        "dataset":          data,
        "history":          hist_a,
        "gradient_history": grad_hist,
        "y_pred":           y_pred_a,
        "y_true_test":      data["y_test"],
        "metrics":          metrics_a,
        "model":            model_a,
    }

    # ── Comparison mode — always BCE vs Focal so students see the contrast ────
    if ss.comparison_mode:
        alt_loss  = "Binary Cross Entropy" if ss.loss_name == "Focal Loss" else "Focal Loss"
        model_b   = build(alt_loss)
        trainer_b = Trainer(model_b, data["X_train"], data["y_train"])
        hist_b, _ = trainer_b.train(
            epochs           = ss.epochs,
            batch_size       = ss.batch_size,
            progress_fn      = prog_fn,
            status_fn        = status_fn,
            record_gradients = False,
            class_weight     = class_weight_dict,   # ROOT CAUSE 1 for model B too
        )
        y_pred_b = trainer_b.predict(data["X_test"])

        # ROOT CAUSE 3: separate optimal threshold for model B
        best_thr_b = best_threshold_from_proba(
            data["y_test"], y_pred_b.ravel(), pos_label=1
        )
        st.session_state.exp4_best_threshold_b = best_thr_b

        metrics_b = compute_metrics(
            data["y_test"], y_pred_b,
            task_type = "binary",
            threshold = best_thr_b,
        )
        results.update({
            "history_b": hist_b,
            "y_pred_b":  y_pred_b,
            "metrics_b": metrics_b,
            "model_b":   model_b,
        })

    return results


# Helper for build_mlp to accept focal params
def _patch_build_mlp():
    """Monkey-patch build_mlp to pass focal params through loss_registry."""
    pass   # Handled via get_loss signature in loss_registry.py


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 5 — Autoencoder Reconstruction (4 loss modes)
#
# LOSS MODES:
#   "MSE Reconstruction"  → standard Conv AE, compiled with loss="mse"
#   "BCE Reconstruction"  → standard Conv AE, compiled with loss="binary_crossentropy"
#   "VAE Loss (Recon+KL)" → Variational AE, custom train_step (NO loss= in compile)
#   "Denoising AE (MSE)"  → standard Conv AE trained on (noisy, clean) pairs
# ─────────────────────────────────────────────────────────────────────────────
def _train_exp5(prog_fn, status_fn) -> dict:
    from datasets.mnist_loader import load_mnist
    from models.autoencoder import (
        build_autoencoder,
        build_and_compile_vae,
        build_denoising_ae,
        add_noise,
    )
    from training.trainer import Trainer
    from utils.metrics import compute_metrics

    ss        = st.session_state
    loss_mode = ss.loss_name         # one of the 4 strings above
    latent_dim = ss.latent_dim       # from sidebar slider
    data       = load_mnist(flatten=False, subset=8000)

    X_tr = data["X_train"]           # (N, 28, 28, 1) float32 in [0,1]
    X_te = data["X_test"]

    # Store active mode for tab rendering
    st.session_state.exp5_mode = loss_mode

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1 & 2: Standard Autoencoder with MSE or BCE reconstruction
    # ─────────────────────────────────────────────────────────────────────────
    if loss_mode in ("MSE Reconstruction", "BCE Reconstruction"):
        ae_loss = "MSE" if loss_mode == "MSE Reconstruction" else "BCE"
        autoencoder, encoder, decoder = build_autoencoder(
            input_shape    = (28, 28, 1),
            latent_dim     = latent_dim,
            learning_rate  = ss.learning_rate,
            optimizer_name = ss.optimizer,
            loss_name      = ae_loss,
        )
        # For AE: input = target (unsupervised reconstruction)
        trainer = Trainer(autoencoder, X_tr, X_tr, validation_split=0.1)
        hist, grad_hist = trainer.train(
            epochs    = ss.epochs,
            batch_size= ss.batch_size,
            progress_fn = prog_fn,
            status_fn   = status_fn,
            # GradientRecorder will resolve "mse"/"binary_crossentropy" → callable
            # via _resolve_loss_callable() — the TypeError bug is fixed.
        )
        reconstructed = autoencoder.predict(X_te, verbose=0)
        metrics       = compute_metrics(X_te, reconstructed, task_type="reconstruction")

        # Latent codes for scatter plot
        latent_codes  = encoder.predict(X_te, verbose=0)
        latent_labels = data["y_test"]

        return {
            "dataset":          data,
            "history":          hist,
            "gradient_history": grad_hist,
            "y_pred":           reconstructed,
            "y_true_test":      X_te,
            "metrics":          metrics,
            "model":            autoencoder,
            "encoder":          encoder,
            "latent_codes":     latent_codes,
            "latent_labels":    latent_labels,
            "reconstructed":    reconstructed,
            "vae_kl_history":   None,
            "vae_recon_history":None,
            "X_train_noisy":    None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3: Variational Autoencoder (VAE Loss = Recon + β·KL)
    #
    # CRITICAL: vae.compile() is called WITHOUT loss=.
    # The ELBO is computed inside VAE.train_step / VAE.test_step.
    # DO NOT add loss= to vae.compile() — it will conflict with train_step.
    # ─────────────────────────────────────────────────────────────────────────
    elif loss_mode == "VAE Loss (Recon + KL)":
        beta = ss.vae_beta
        vae, encoder, decoder = build_and_compile_vae(
            input_shape    = (28, 28, 1),
            latent_dim     = latent_dim,
            beta           = beta,
            learning_rate  = ss.learning_rate,
            optimizer_name = ss.optimizer,
        )

        # VAE uses custom train_step → model.loss is None → GradientRecorder
        # cannot record gradients.  Pass record_gradients=False to skip it
        # gracefully (the fixed Trainer._resolve_loss_callable handles this).
        trainer = Trainer(vae, X_tr, X_tr, validation_split=0.1)
        hist, grad_hist = trainer.train(
            epochs          = ss.epochs,
            batch_size      = ss.batch_size,
            progress_fn     = prog_fn,
            status_fn       = status_fn,
            record_gradients= False,   # VAE has no model.loss — skip gracefully
        )

        # VAE reconstruction: encode → sample z → decode
        z_mean_all, z_log_var_all, z_all = encoder.predict(X_te, verbose=0)
        reconstructed = decoder.predict(z_mean_all, verbose=0)   # use mean, not sample
        metrics       = compute_metrics(X_te, reconstructed, task_type="reconstruction")

        # Latent codes (use z_mean for deterministic scatter plot)
        latent_codes  = z_mean_all
        latent_labels = data["y_test"]

        # Store split histories so the training dashboard can plot 3 curves
        st.session_state.vae_kl_history    = hist.get("kl_loss", [])
        st.session_state.vae_recon_history = hist.get("recon_loss", [])

        return {
            "dataset":          data,
            "history":          hist,        # has "total_loss", "recon_loss", "kl_loss"
            "gradient_history": {},          # empty — VAE gradients not recorded
            "y_pred":           reconstructed,
            "y_true_test":      X_te,
            "metrics":          metrics,
            "model":            vae,
            "encoder":          encoder,
            "latent_codes":     latent_codes,
            "latent_labels":    latent_labels,
            "reconstructed":    reconstructed,
            "vae_kl_history":   hist.get("kl_loss", []),
            "vae_recon_history":hist.get("recon_loss", []),
            "X_train_noisy":    None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 4: Denoising Autoencoder
    # Input  = clean + Gaussian noise     (X_train_noisy → encoder)
    # Target = clean original             (X_train_clean → loss computation)
    # The model must learn to remove noise → extract signal, not noise identity
    # ─────────────────────────────────────────────────────────────────────────
    elif loss_mode == "Denoising AE (MSE)":
        noise_factor = ss.noise_factor
        autoencoder, encoder, decoder = build_denoising_ae(
            input_shape    = (28, 28, 1),
            latent_dim     = latent_dim,
            learning_rate  = ss.learning_rate,
            optimizer_name = ss.optimizer,
        )

        # Add noise to training inputs; keep clean as targets
        X_tr_noisy = add_noise(X_tr, noise_factor=noise_factor)
        X_te_noisy = add_noise(X_te, noise_factor=noise_factor)

        # Trainer: input=noisy, target=clean (MSE between reconstruction and clean)
        trainer = Trainer(autoencoder, X_tr_noisy, X_tr, validation_split=0.1)
        hist, grad_hist = trainer.train(
            epochs    = ss.epochs,
            batch_size= ss.batch_size,
            progress_fn = prog_fn,
            status_fn   = status_fn,
        )

        # Evaluate: noisy → reconstructed; compare to clean original
        reconstructed = autoencoder.predict(X_te_noisy, verbose=0)
        metrics       = compute_metrics(X_te, reconstructed, task_type="reconstruction")

        latent_codes  = encoder.predict(X_te_noisy, verbose=0)
        latent_labels = data["y_test"]

        return {
            "dataset":          data,
            "history":          hist,
            "gradient_history": grad_hist,
            "y_pred":           reconstructed,
            "y_true_test":      X_te,          # clean originals
            "metrics":          metrics,
            "model":            autoencoder,
            "encoder":          encoder,
            "latent_codes":     latent_codes,
            "latent_labels":    latent_labels,
            "reconstructed":    reconstructed,
            "vae_kl_history":   None,
            "vae_recon_history":None,
            "X_train_noisy":    X_te_noisy,    # for the 3-column viz (noisy/recon/clean)
        }

    # Fallback (should never reach here if loss options are set correctly in config)
    raise ValueError(f"Unknown Experiment 5 loss mode: {loss_mode}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE TABS
# ═════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the main page header."""
    exp = EXPERIMENTS[st.session_state.experiment_id]
    st.markdown(f"""
    <div style="padding:20px 0 10px 0;">
        <h1 style="font-size:2rem; font-weight:800; margin:0;
                   background:linear-gradient(135deg,#4A90D9,#7EC8E3);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🧪 Interactive Loss Function Virtual Lab
        </h1>
        <p style="color:#A0A8C0; margin:4px 0 0 2px; font-size:0.95rem;">
            {exp['icon']} Experiment {st.session_state.experiment_id}: 
            <b style="color:#7EC8E3;">{exp['name']}</b> 
            &nbsp;·&nbsp; Dataset: {exp['dataset']}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(metrics: dict):
    """Render metrics as styled cards in a responsive column grid."""
    if not metrics:
        return
    cols = st.columns(min(len(metrics), 5))
    for i, (name, val) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            if isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{formatted}</div>
                <div class="metric-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 1: Training Dashboard ─────────────────────────────────────────────────
def tab_training_dashboard():
    ss      = st.session_state
    exp     = EXPERIMENTS[ss.experiment_id]
    loss_nm = ss.loss_name
    meta    = LOSS_METADATA.get(loss_nm, {})

    st.markdown('<div class="section-header">📐 Loss Formula</div>', unsafe_allow_html=True)
    if meta.get("latex"):
        st.latex(meta["latex"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Description**")
        st.markdown(f'<div class="info-box">{meta.get("description", "")}</div>',
                    unsafe_allow_html=True)

    with col2:
        if not ss.trained:
            st.info(
                "👈 Configure your experiment in the sidebar and click **🚀 TRAIN MODEL** to begin.",
                icon="💡",
            )
            return

        # ── Metrics table ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Final Test Metrics</div>',
                    unsafe_allow_html=True)
        if ss.metrics:
            render_metric_cards(ss.metrics)

    if not ss.trained:
        return

    from plots.loss_curves import plot_loss_curves

    st.divider()
    st.markdown('<div class="section-header">📈 Training History</div>',
                unsafe_allow_html=True)

    task_type = exp["task_type"]

    # ── Special VAE training history: 3 curves (Total, Recon, KL) ─────────
    if ss.experiment_id == 5 and ss.exp5_mode == "VAE Loss (Recon + KL)":
        import plotly.graph_objects as go
        hist = ss.history
        epochs_x = list(range(1, len(hist.get("total_loss", [])) + 1))

        fig_vae = go.Figure()
        if "total_loss" in hist:
            fig_vae.add_trace(go.Scatter(
                x=epochs_x, y=hist["total_loss"],
                mode="lines", name="Total Loss (ELBO)",
                line=dict(color=COLORS["loss_accent"], width=2.5),
            ))
        if "recon_loss" in hist:
            fig_vae.add_trace(go.Scatter(
                x=epochs_x, y=hist["recon_loss"],
                mode="lines", name="Reconstruction Loss",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
            ))
        if "kl_loss" in hist:
            fig_vae.add_trace(go.Scatter(
                x=epochs_x, y=hist["kl_loss"],
                mode="lines", name="KL Divergence",
                line=dict(color=COLORS["accuracy"], width=2, dash="dot"),
            ))
        fig_vae.update_layout(
            template="plotly_dark",
            title=f"VAE Training — Total = Recon + β·KL  (β={ss.vae_beta:.1f})",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=40, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_vae, use_container_width=True, key="dashboard_vae_history_1")
        st.caption(
            "💡 **Reading the curves**: KL loss rises early (encoder learns structured posterior), "
            "Recon loss falls (decoder gets better). Stable convergence = Total loss smoothly descending."
        )
    else:
        fig = plot_loss_curves(ss.history, title=f"{loss_nm} — Training History",
                               task_type=task_type)
        st.plotly_chart(fig, use_container_width=True, key="dashboard_training_history_1")

    # ── Dataset statistics summary ─────────────────────────────────────────
    st.divider()
    _render_dataset_summary()


def _render_dataset_summary():
    ss = st.session_state
    if not ss.dataset:
        return

    exp_id = ss.experiment_id
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📋 Dataset Info</div>',
                    unsafe_allow_html=True)
        data = ss.dataset
        info = {}

        if exp_id == 1:
            info = {
                "Train samples": len(data["X_train_flat"]),
                "Test samples":  len(data["X_test_flat"]),
                "Features":  784,
                "Classes":   10,
                "Shape":     "28×28×1 (MNIST)",
            }
        elif exp_id == 2:
            info = {
                "Train samples": len(data["X_train"]),
                "Test samples":  len(data["X_test"]),
                "Features":  data["n_features"],
                "Classes":   ["Malignant", "Benign"],
                "Dataset":   "Breast Cancer (sklearn)",
            }
        elif exp_id == 3:
            info = {
                "Train samples": len(data["X_train"]),
                "Test samples":  len(data["X_test"]),
                "Features":  data["n_features"],
                "Outliers":  data["n_outliers"],
                "Dataset":   "California Housing",
            }
        elif exp_id == 4:
            info = {
                "Train samples": len(data["X_train"]),
                "Fraud (train)": int(data["y_train"].sum()),
                "Normal (train)": int((1 - data["y_train"]).sum()),
                "Fraud ratio":   f"{data['fraud_ratio']:.1%}",
                "Features":  data["n_features"],
            }
        elif exp_id == 5:
            mode  = ss.get("exp5_mode", ss.loss_name)
            info  = {
                "Train samples": len(data["X_train"]),
                "Test samples":  len(data["X_test"]),
                "Input shape":   "28×28×1",
                "Latent dim":    ss.latent_dim,
                "Loss mode":     mode,
                "Dataset":       "MNIST (autoencoder)",
            }
            # VAE-specific extras
            if "VAE" in mode:
                info["β (beta)"] = ss.vae_beta
            if "Denoising" in mode:
                info["Noise factor (σ)"] = ss.noise_factor

        df = pd.DataFrame({"Property": list(info.keys()),
                           "Value": [str(v) for v in info.values()]})
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col2:
        if ss.show_dataset:
            _render_dataset_preview()


def _render_dataset_preview():
    """Show a quick dataset preview depending on experiment type."""
    ss     = st.session_state
    exp_id = ss.experiment_id
    data   = ss.dataset

    st.markdown('<div class="section-header">🔍 Dataset Preview</div>',
                unsafe_allow_html=True)

    if exp_id in (1, 5):
        # Show first 9 MNIST images
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        X = data["X_train"][:9, ..., 0]   # (9, 28, 28)
        y = data["y_train"][:9]

        fig = make_subplots(rows=3, cols=3,
                            subplot_titles=[f"Label: {lbl}" for lbl in y],
                            horizontal_spacing=0.04, vertical_spacing=0.1)
        for idx in range(9):
            r, c = idx // 3 + 1, idx % 3 + 1
            img_uint8 = (X[idx] * 255).astype(np.uint8)
            img_rgb   = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
            fig.add_trace(go.Image(z=img_rgb, hoverinfo="none"), row=r, col=c)

        fig.update_layout(
            template="plotly_dark", height=320,
            margin=dict(t=30, b=5, l=5, r=5),
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(fig, use_container_width=True, key="dataset_preview_mnist_grid_1")

    elif exp_id == 4:
        # Class distribution bar chart
        import plotly.graph_objects as go
        counts = data["class_counts"]
        fig = go.Figure(go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[COLORS["accuracy"], COLORS["loss_accent"]],
            text=list(counts.values()),
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", height=250,
            title="Class Distribution",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="dataset_preview_class_dist_1")
        imb = data["fraud_ratio"]
        st.markdown(
            f'<span class="winner-badge" '
            f'style="background:linear-gradient(135deg,#FF6B6B,#CC2929);">'
            f'⚠️ Imbalance: {imb:.1%} fraud</span>',
            unsafe_allow_html=True,
        )

    elif exp_id == 3:
        # Scatter: first feature vs target with outliers highlighted
        import plotly.graph_objects as go
        X = data["X_train"]
        y = data["y_train"]
        y_clean = data["y_train_clean"]
        is_out  = np.abs(y - y_clean) > 0.01

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[~is_out, 0], y=y[~is_out],
            mode="markers", marker=dict(size=3, color=COLORS["primary"], opacity=0.6),
            name="Normal",
        ))
        if is_out.any():
            fig.add_trace(go.Scatter(
                x=X[is_out, 0], y=y[is_out],
                mode="markers", marker=dict(size=6, color=COLORS["loss_accent"],
                                             symbol="x", opacity=0.9),
                name="Outliers",
            ))
        fig.update_layout(
            template="plotly_dark", height=250,
            title="Feature 0 vs Target (with outliers)",
            xaxis_title="Feature 0 (scaled)", yaxis_title="House Value",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="dataset_preview_regression_scatter_1")


# ── TAB 2: Predictions ────────────────────────────────────────────────────────
def tab_predictions():
    ss = st.session_state

    if not ss.trained:
        st.info("Train a model first using the sidebar.", icon="⏳")
        return

    exp_id = ss.experiment_id
    exp    = EXPERIMENTS[exp_id]

    if exp_id == 1:
        from plots.prediction_viz import plot_mnist_grid
        n_show = st.slider("Images to display", 9, 25, 25, step=4)
        fig = plot_mnist_grid(
            X_images    = ss.dataset["X_test_flat"] if ss.model_type == "MLP"
                          else ss.dataset["X_test"],
            y_true       = ss.y_true_test,
            y_pred_probs = ss.y_pred,
            n_show       = n_show,
        )
        st.plotly_chart(fig, use_container_width=True, key="predictions_mnist_grid_1")

        # Per-class accuracy
        st.markdown("**Per-class Accuracy**")
        y_pred_cls = np.argmax(ss.y_pred, axis=1)
        y_true_cls = ss.y_true_test.astype(int)
        per_class  = {}
        for c in range(10):
            mask = y_true_cls == c
            if mask.sum() > 0:
                per_class[str(c)] = float((y_pred_cls[mask] == c).mean())

        import plotly.graph_objects as go
        fig2 = go.Figure(go.Bar(
            x=list(per_class.keys()),
            y=list(per_class.values()),
            marker_color=[
                COLORS["accuracy"] if v >= 0.9 else COLORS["val_accent"] if v >= 0.7
                else COLORS["loss_accent"]
                for v in per_class.values()
            ],
            text=[f"{v:.1%}" for v in per_class.values()],
            textposition="outside",
        ))
        fig2.update_layout(
            template="plotly_dark", height=300,
            title="Per-Digit Classification Accuracy",
            xaxis_title="Digit", yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.1]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=50, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True, key="predictions_per_class_accuracy_1")

    elif exp_id == 2:
        from plots.prediction_viz import plot_binary_probabilities

        threshold = ss.threshold
        fig = plot_binary_probabilities(
            y_true       = ss.y_true_test,
            y_pred_probs = ss.y_pred.ravel(),
            threshold    = threshold,
            n_show       = 60,
        )
        st.plotly_chart(fig, use_container_width=True, key="predictions_binary_probs_1")

        st.caption(
            f"🎯 Current threshold: **{threshold:.2f}** — "
            f"adjust via the sidebar slider without re-training."
        )

        # Live metrics at current threshold
        from utils.metrics import compute_metrics
        live_metrics = compute_metrics(
            ss.y_true_test, ss.y_pred,
            task_type="binary", threshold=threshold,
        )
        render_metric_cards(live_metrics)

    elif exp_id == 3:
        from plots.prediction_viz import plot_regression_predictions

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📊 {ss.loss_name} Predictions**")
            fig_a = plot_regression_predictions(
                ss.y_true_test, ss.y_pred.ravel(),
                loss_name=ss.loss_name,
            )
            st.plotly_chart(fig_a, use_container_width=True, key="predictions_regression_a_1")

        with col2:
            if ss.comparison_mode and ss.y_pred_b is not None:
                alt_loss = "Huber" if ss.loss_name == "MSE" else "MSE"
                st.markdown(f"**📊 {alt_loss} Predictions**")
                fig_b = plot_regression_predictions(
                    ss.y_true_test, ss.y_pred_b.ravel(),
                    loss_name=alt_loss,
                )
                st.plotly_chart(fig_b, use_container_width=True, key="predictions_regression_b_1")
            else:
                st.info("Enable Comparison Mode in the sidebar to see side-by-side predictions.")

    elif exp_id == 4:
        from plots.prediction_viz import plot_binary_probabilities
        from utils.metrics import compute_metrics, best_threshold_from_proba
        from sklearn.metrics import confusion_matrix as sk_cm
        import plotly.graph_objects as go

        # ── UI IMPROVEMENT 3: colour-coded F1 health badge ────────────────────
        f1    = ss.metrics.get("F1 Score", 0.0)
        rec   = ss.metrics.get("Recall",   0.0)
        prec  = ss.metrics.get("Precision", 0.0)
        auc   = ss.metrics.get("ROC-AUC",  None)
        thr   = ss.threshold

        # UI IMPROVEMENT 1: model collapse warning BEFORE any charts
        if rec < 0.01:
            st.error(
                "⚠️ **Model collapse detected** — the fraud class is never predicted!  \n"
                "All samples are classified as *normal* (class 0).  \n\n"
                "**Why this happens:** The model found a 'lazy shortcut' → "
                "predicting all-normal gives ~95% accuracy when fraud is only 5%.  \n\n"
                "**Try:** Lower the threshold slider (sidebar) · Increase epochs · "
                "Switch to **Focal Loss** with high γ.",
                icon="🚨",
            )
        elif f1 < 0.2:
            st.warning(
                f"⚡ F1 Score = {f1:.3f} — fraud detection is weak.  \n"
                "Try lowering the threshold or switching to **Focal Loss**.",
                icon="⚠️",
            )

        # UI IMPROVEMENT 2: show class distribution before charts
        fraud_pct = float(ss.y_true_test.mean() * 100)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Fraud % in test set", f"{fraud_pct:.1f}%",
                      help="Minority class fraction — drives threshold sensitivity")
            st.progress(min(fraud_pct / 100, 1.0))
        with c2:
            # UI IMPROVEMENT 3: colour the F1 card by health
            if f1 >= 0.5:
                f1_color = "#52B788"   # green — good
                f1_label = "✅ F1 (Fraud)"
            elif f1 >= 0.2:
                f1_color = "#F4A261"   # orange — marginal
                f1_label = "⚡ F1 (Fraud)"
            else:
                f1_color = "#FF6B6B"   # red — collapsed
                f1_label = "🚨 F1 (Fraud)"
            st.markdown(
                f'<div class="metric-card" style="border:2px solid {f1_color};">'
                f'<div class="metric-value" style="color:{f1_color};">{f1:.4f}</div>'
                f'<div class="metric-label">{f1_label}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            st.metric("ROC-AUC", auc_str,
                      help="Area under ROC curve — threshold-independent quality measure")

        st.divider()

        # Live probability distribution chart
        fig = plot_binary_probabilities(
            y_true       = ss.y_true_test,
            y_pred_probs = ss.y_pred.ravel(),
            threshold    = thr,
            title        = f"{ss.loss_name} — Fraud Probability Scores (threshold={thr:.2f})",
        )
        st.plotly_chart(fig, use_container_width=True, key="predictions_fraud_probs_1")

        st.caption(
            f"🎯 Current threshold: **{thr:.3f}** "
            f"(auto-detected best = **{ss.exp4_best_threshold:.3f}**) — "
            "adjust via the **sidebar slider** without re-training."
        )

        # UI IMPROVEMENT 5: Fraud caught vs missed in absolute counts
        st.divider()
        st.markdown("#### 🎯 Fraud Detection Count (Absolute)")
        y_bin = (ss.y_pred.ravel() >= thr).astype(int)
        y_true_int = ss.y_true_test.astype(int)
        cm = sk_cm(y_true_int, y_bin, labels=[0, 1])
        tp = int(cm[1, 1])   # fraud caught
        fn = int(cm[1, 0])   # fraud missed
        fp = int(cm[0, 1])   # false alarms
        tn = int(cm[0, 0])   # correct normal
        total_fraud = tp + fn

        cnt_col1, cnt_col2, cnt_col3 = st.columns(3)
        with cnt_col1:
            st.metric(
                label="✅ True Fraud Caught",
                value=f"{tp} / {total_fraud}",
                delta=f"{tp/max(total_fraud,1):.0%} recall",
                delta_color="normal",
            )
        with cnt_col2:
            st.metric(
                label="❌ Fraud Missed",
                value=f"{fn} / {total_fraud}",
                delta=f"{fn/max(total_fraud,1):.0%} miss rate",
                delta_color="inverse",
            )
        with cnt_col3:
            st.metric(
                label="⚠️ False Alarms",
                value=fp,
                help="Normal transactions flagged as fraud",
            )

        st.divider()

        # UI IMPROVEMENT 4: show accuracy paradox with live metrics
        # This IS the core lesson: compare your model to a "predict all normal" dummy
        acc       = ss.metrics.get("Accuracy", 0)
        dummy_acc = 1.0 - ss.fraud_ratio

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Accuracy",
            x=["Dummy (predict all normal)", f"{ss.loss_name}"],
            y=[dummy_acc, acc],
            marker_color=COLORS["accuracy"],
            text=[f"{dummy_acc:.2%}", f"{acc:.2%}"],
            textposition="outside",
        ))
        fig2.add_trace(go.Bar(
            name="F1 Score (Fraud class)",
            x=["Dummy (predict all normal)", f"{ss.loss_name}"],
            y=[0.0, f1],
            marker_color=COLORS["loss_accent"],
            text=["0.00 ← model collapse!", f"{f1:.4f}"],
            textposition="outside",
        ))
        fig2.update_layout(
            barmode="group",
            template="plotly_dark",
            title="⚖️ Accuracy / F1 Paradox — High Accuracy ≠ Good Fraud Detection",
            height=340,
            yaxis=dict(range=[0, 1.15]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=40, l=60, r=20),
        )
        fig2.add_annotation(
            text=(
                "💡 The Dummy model gets 93% accuracy by predicting ALL normal.  "
                "F1=0 reveals it never detects fraud."
            ),
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11, color="#A0A8C0"),
        )
        st.plotly_chart(fig2, use_container_width=True, key="predictions_accuracy_paradox_1")

        # Live metric cards at current threshold (updates with sidebar slider)
        st.markdown("#### 📋 Live Metrics at Current Threshold")
        live_metrics = compute_metrics(
            ss.y_true_test, ss.y_pred, task_type="binary", threshold=thr
        )
        render_metric_cards(live_metrics)

    elif exp_id == 5:
        from plots.prediction_viz import plot_autoencoder_reconstructions

        mode = ss.get("exp5_mode", ss.loss_name)

        # ── Reconstruction metrics ─────────────────────────────────────────
        render_metric_cards(ss.metrics)
        st.divider()

        # ── DENOISING AE: 3-column layout (Noisy | Reconstruction | Clean) ──
        if mode == "Denoising AE (MSE)" and ss.get("X_train_noisy") is not None:
            st.markdown("#### 🔇 Denoising: Noisy Input → Reconstruction → Clean Original")
            n_show = st.slider("Images to show", 4, 10, 6, key="denoise_n_show")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔊 Noisy Input**")
                fig_noisy = plot_autoencoder_reconstructions(
                    X_original      = ss.X_train_noisy,
                    X_reconstructed = ss.X_train_noisy,
                    n_pairs         = n_show,
                )
                st.plotly_chart(fig_noisy, use_container_width=True,
                                key="predictions_ae_denoise_noisy_1")
            with col2:
                st.markdown("**🔧 Reconstructed (Denoised)**")
                fig_recon = plot_autoencoder_reconstructions(
                    X_original      = ss.y_pred,
                    X_reconstructed = ss.y_pred,
                    n_pairs         = n_show,
                )
                st.plotly_chart(fig_recon, use_container_width=True,
                                key="predictions_ae_denoise_recon_1")
            with col3:
                st.markdown("**✅ Clean Original**")
                fig_clean = plot_autoencoder_reconstructions(
                    X_original      = ss.y_true_test,
                    X_reconstructed = ss.y_true_test,
                    n_pairs         = n_show,
                )
                st.plotly_chart(fig_clean, use_container_width=True,
                                key="predictions_ae_denoise_clean_1")

            st.caption(
                "💡 **Interpretation**: The denoiser must separate 'signal' (digit structure) "
                "from 'noise' (random Gaussian). Compare column 1 (corrupted input) vs "
                "column 2 (recovered digit) — clear improvement means the AE learned "
                "noise-invariant features."
            )

        # ── VAE: Reconstruction grid + Generation section ──────────────────
        elif mode == "VAE Loss (Recon + KL)":
            st.markdown("#### 🔄 Section A: Reconstructions (test set → encode → decode)")
            n_pairs = st.slider("Image pairs to show", 4, 10, 8, key="vae_recon_n")
            fig_r = plot_autoencoder_reconstructions(
                X_original      = ss.y_true_test,
                X_reconstructed = ss.reconstructed,
                n_pairs         = n_pairs,
            )
            st.plotly_chart(fig_r, use_container_width=True,
                            key="predictions_vae_recon_1")

            st.divider()
            st.markdown(
                "#### ✨ Section B: Generation — Sampling z ~ N(0, I) → Decoder"
            )
            st.markdown(
                '<div class="info-box">'
                "These digits were <b>GENERATED</b>, not reconstructed. "
                "Random latent vectors z ~ N(0,I) are sampled and passed directly through "
                "the decoder — possible only because the VAE's KL term regularises the "
                "latent space to approximate a unit Gaussian."
                "</div>",
                unsafe_allow_html=True,
            )
            n_gen = st.slider("Digits to generate", 5, 20, 10, key="vae_gen_n")
            _render_vae_generation(n_gen)

        # ── MSE / BCE: standard side-by-side reconstruction grid ───────────
        else:
            n_pairs = st.slider("Image pairs to show", 4, 10, 8, key="ae_recon_n")
            fig = plot_autoencoder_reconstructions(
                X_original      = ss.y_true_test,
                X_reconstructed = ss.reconstructed,
                n_pairs         = n_pairs,
            )
            st.plotly_chart(fig, use_container_width=True,
                            key="predictions_ae_reconstructions_1")

        # ── Latent space scatter (all modes) ──────────────────────────────
        if ss.latent_codes is not None:
            st.divider()
            # FIX-3: pass unique key_prefix for Tab 2 call site
            _render_latent_space(key_prefix="tab2_predictions")


def _render_vae_generation(n_gen: int = 10):
    """Sample z ~ N(0,I), decode, display generated digits."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    ss = st.session_state
    decoder = ss.get("encoder")   # we don't store decoder separately; use model

    # Access decoder from the VAE model (which stores encoder + decoder)
    vae_model = ss.model
    if not hasattr(vae_model, "decoder"):
        st.info("Generation requires a VAE model with a .decoder attribute.")
        return

    decoder    = vae_model.decoder
    latent_dim = ss.latent_codes.shape[1] if ss.latent_codes is not None else 8

    # Sample random points from the prior N(0,I)
    z_samples  = np.random.normal(0, 1, size=(n_gen, latent_dim)).astype("float32")
    generated  = decoder.predict(z_samples, verbose=0)   # (n_gen, 28, 28, 1)

    # Build subplot grid
    cols   = min(n_gen, 5)
    rows   = (n_gen + cols - 1) // cols
    titles = [f"Gen {i+1}" for i in range(n_gen)]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.02, vertical_spacing=0.08)
    for idx in range(n_gen):
        r, c    = idx // cols + 1, idx % cols + 1
        img     = generated[idx, ..., 0]
        img_u8  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
        fig.add_trace(go.Image(z=img_rgb, hoverinfo="none"), row=r, col=c)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(showticklabels=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"🎲 VAE-Generated Digits (z ~ N(0,I), latent_dim={latent_dim})",
        height=rows * 130 + 80,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="predictions_vae_generation_1")


# FIX-3: Added key_prefix parameter so each call site generates a unique key.
# Called from tab_predictions() (Tab 2) and tab_loss_analysis() (Tab 3).
def _render_latent_space(key_prefix: str = "tab2_predictions"):
    """
    Latent space scatter coloured by digit label.
    If latent_dim == 2: direct 2D scatter.
    If latent_dim > 2: reduce to 2D with PCA first, annotate as 'PCA projection'.

    Parameters
    ----------
    key_prefix : unique string prefix for the plotly_chart key, e.g.
                 "tab2_predictions" or "tab3_loss_analysis".
                 Must differ across all call sites to avoid DuplicateElementKey.
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    import numpy as np

    ss     = st.session_state
    codes  = ss.latent_codes
    labels = ss.latent_labels

    if codes is None or len(codes) == 0:
        return

    # Dimensionality reduction if needed
    if codes.shape[1] == 2:
        z2d  = codes
        ax_label_prefix = "Latent Dim"
        title_suffix    = "(2D latent space)"
    else:
        pca   = PCA(n_components=2)
        z2d   = pca.fit_transform(codes)
        ax_label_prefix = "PCA Component"
        var_exp = pca.explained_variance_ratio_ * 100
        title_suffix = (
            f"(PCA of {codes.shape[1]}D latent space — "
            f"{var_exp[0]:.1f}% + {var_exp[1]:.1f}% variance explained)"
        )

    fig = go.Figure()
    colors_10 = [
        "#4A90D9", "#FF6B6B", "#6BCB77", "#FFD93D",
        "#C77DFF", "#FF9A3C", "#00B4D8", "#F72585",
        "#80EF80", "#FFB347",
    ]
    for digit in range(10):
        mask = labels == digit
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=z2d[mask, 0], y=z2d[mask, 1],
            mode="markers",
            marker=dict(size=4, opacity=0.7, color=colors_10[digit]),
            name=str(digit),
            hovertemplate=f"Digit {digit}<br>z1=%{{x:.2f}}, z2=%{{y:.2f}}<extra></extra>",
        ))

    mode_label = ss.get("exp5_mode", ss.loss_name)
    fig.update_layout(
        template="plotly_dark",
        title=f"🔮 Latent Space — {mode_label} {title_suffix}",
        xaxis_title=f"{ax_label_prefix} 1",
        yaxis_title=f"{ax_label_prefix} 2",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(title="Digit", font=dict(size=10), orientation="h"),
    )
    # FIX-3: key uses caller-supplied prefix — no two call sites share the same key
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{key_prefix}_latent_scatter_1")

    if mode_label == "VAE Loss (Recon + KL)":
        st.caption(
            "💡 **VAE latent space**: The KL term forces the posterior to resemble N(0,I), "
            "producing smooth, overlapping clusters. You can interpolate between digits by "
            "moving smoothly through latent space — try generating z between two cluster centers."
        )
    else:
        st.caption(
            "💡 **Standard AE latent space**: No KL regularisation → clusters may be "
            "scattered with gaps. Interpolating between clusters can produce nonsensical outputs. "
            "Compare with the VAE scatter to see the regularisation effect."
        )


# ── TAB 3: Loss Analysis ─────────────────────────────────────────────────────
def tab_loss_analysis():
    ss = st.session_state

    if not ss.trained:
        st.info("Train a model first using the sidebar.", icon="⏳")
        return

    exp    = EXPERIMENTS[ss.experiment_id]
    task   = exp["task_type"]
    exp_id = ss.experiment_id

    from plots.confusion_matrix import plot_confusion_matrix
    from plots.roc_curve import plot_roc_curve, plot_pr_curve
    from plots.gradient_viz import plot_gradient_magnitudes

    # ── Confusion Matrix & ROC ────────────────────────────────────────────────
    if task in ("multiclass", "binary"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🎯 Confusion Matrix</div>',
                        unsafe_allow_html=True)

            if task == "multiclass":
                class_names = [str(i) for i in range(10)]
            elif exp_id == 2:
                class_names = ss.dataset.get("class_names", ["Class 0", "Class 1"])
            else:
                class_names = ["Normal", "Fraud"]

            fig_cm = plot_confusion_matrix(
                y_true     = ss.y_true_test,
                y_pred     = ss.y_pred,
                class_names= class_names,
                threshold  = ss.threshold,
                normalize  = True,
            )
            st.plotly_chart(fig_cm, use_container_width=True, key="loss_analysis_confusion_matrix_1")

        with col2:
            st.markdown('<div class="section-header">📈 ROC Curve</div>',
                        unsafe_allow_html=True)
            fig_roc = plot_roc_curve(
                y_true      = ss.y_true_test,
                y_pred_probs= ss.y_pred,
                class_names = class_names if task == "multiclass" else None,
            )
            st.plotly_chart(fig_roc, use_container_width=True, key="loss_analysis_roc_curve_1")

        # Precision-Recall Curve
        st.markdown('<div class="section-header">🎯 Precision-Recall Curve</div>',
                    unsafe_allow_html=True)
        fig_pr = plot_pr_curve(
            y_true      = ss.y_true_test,
            y_pred_probs= ss.y_pred,
            class_names = class_names if task == "multiclass" else None,
        )
        st.plotly_chart(fig_pr, use_container_width=True, key="loss_analysis_pr_curve_1")

    elif task == "regression":
        st.markdown('<div class="section-header">📊 Regression Analysis</div>',
                    unsafe_allow_html=True)
        from plots.prediction_viz import plot_regression_predictions
        fig = plot_regression_predictions(
            ss.y_true_test, ss.y_pred.ravel(), loss_name=ss.loss_name,
        )
        st.plotly_chart(fig, use_container_width=True, key="loss_analysis_regression_predictions_1")

    elif task == "reconstruction":
        mode = ss.get("exp5_mode", ss.loss_name)
        import plotly.graph_objects as go

        # ── Per-sample MSE histogram ──────────────────────────────────────────
        st.markdown("#### 📊 Per-Sample Reconstruction MSE Distribution")
        per_sample_mse = np.mean((ss.y_true_test - ss.y_pred)**2, axis=(1, 2, 3))
        fig_mse = go.Figure(go.Histogram(
            x=per_sample_mse, nbinsx=40,
            marker_color=COLORS["primary"],
        ))
        fig_mse.update_layout(
            template="plotly_dark", height=280,
            title="Per-Sample Reconstruction MSE — lower and tighter = better",
            xaxis_title="MSE per image", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_mse, use_container_width=True, key="loss_analysis_recon_mse_hist_1")

        # ── VAE extras: latent scatter + KL per dimension ──────────────────
        if mode == "VAE Loss (Recon + KL)":
            st.divider()
            st.markdown("#### 🔮 Latent Space Scatter")
            if ss.latent_codes is not None:
                # FIX-3: pass unique key_prefix for Tab 3 call site
                _render_latent_space(key_prefix="tab3_loss_analysis")

            st.divider()
            st.markdown(
                "#### 📊 KL Divergence per Latent Dimension\n"
                "Shows which latent dimensions are actively used (high KL) vs "
                "collapsed to the prior (KL ≈ 0)."
            )
            # Compute KL per dim from the latent codes
            # We need z_mean and z_log_var — encode a batch of test images
            if hasattr(ss.model, "encoder"):
                X_sample = ss.dataset["X_test"][:256]
                z_mean, z_log_var, _ = ss.model.encoder.predict(X_sample, verbose=0)
                # KL_i = -0.5 * (1 + log_var_i - mu_i^2 - exp(log_var_i))
                kl_per_dim = -0.5 * np.mean(
                    1 + z_log_var - np.square(z_mean) - np.exp(z_log_var),
                    axis=0
                )  # (latent_dim,)
                dims = [f"z{i}" for i in range(len(kl_per_dim))]
                # Color: high KL = active (blue), near-zero = collapsed (red)
                colors_kl = [
                    COLORS["primary"] if v > 0.05 else COLORS["loss_accent"]
                    for v in kl_per_dim
                ]
                fig_kl = go.Figure(go.Bar(
                    x=dims, y=kl_per_dim,
                    marker_color=colors_kl,
                    text=[f"{v:.3f}" for v in kl_per_dim],
                    textposition="outside",
                ))
                fig_kl.update_layout(
                    template="plotly_dark", height=320,
                    title=(f"KL per Latent Dimension — β={ss.vae_beta:.1f}  "
                           f"(blue=active, red=collapsed)"),
                    xaxis_title="Latent Dimension", yaxis_title="KL Divergence",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=60, b=40, l=60, r=20),
                )
                fig_kl.add_hline(
                    y=0.05, line_dash="dash", line_color="#FFD93D",
                    annotation_text="KL < 0.05 = collapsed dim",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_kl, use_container_width=True,
                                key="loss_analysis_vae_kl_per_dim_1")
                st.caption(
                    "💡 **Interpretation**: Active dims (blue) encode actual digit features. "
                    "Collapsed dims (red) are unused — the encoder sends them to z~N(0,1) automatically. "
                    f"Increase β to force more dims to collapse (more disentangled); "
                    f"decrease β to use more dims (sharper reconstructions)."
                )
            else:
                st.info("KL-per-dim chart requires a VAE with .encoder attribute.")

    # ── Gradient Magnitude ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🔬 Gradient Magnitudes per Layer</div>',
                unsafe_allow_html=True)
    if ss.gradient_history:
        fig_grad = plot_gradient_magnitudes(ss.gradient_history)
        st.plotly_chart(fig_grad, use_container_width=True, key="loss_analysis_gradient_magnitudes_1")
        st.caption(
            "💡 **Reading the chart**: Values near zero = vanishing gradients "
            "(layers learn very slowly). Spikes = large gradient steps. "
            "A healthy net shows smoothly decreasing norms across epochs."
        )
    else:
        st.info("No gradient history available for this model type.")


# ── TAB 4: Loss Landscape ─────────────────────────────────────────────────────
def tab_loss_landscape():
    ss = st.session_state

    if not ss.trained:
        st.info("Train a model first to generate the loss landscape.", icon="⏳")
        return

    # ── FIX-2 (Part B): Exp5 guard — autoencoder losses cannot produce a
    # meaningful 2D weight-perturbation landscape (VAE/reconstruction inputs
    # are 28×28 images, not tabular; VAE loss has no simple scalar form).
    _EXP5_LOSS_NAMES = {
        "mse reconstruction", "bce reconstruction",
        "vae loss (recon + kl)", "vae loss (recon+kl)", "denoising ae (mse)",
    }
    _current_loss_lower = (ss.get("loss_name") or "").strip().lower()

    if _current_loss_lower in _EXP5_LOSS_NAMES or ss.get("experiment_id") == 5:
        st.info(
            "🌋 **Loss Landscape** is not available for Autoencoder experiments.  \n"
            "The VAE and reconstruction losses operate on high-dimensional image "
            "inputs — the 2D weight perturbation grid is only meaningful for "
            "tabular classification/regression tasks.",
            icon="ℹ️",
        )
        st.markdown("""
**Why?**
- VAE loss = Recon + KL — cannot be reduced to a simple 2-weight surface
- Autoencoder inputs are 28×28 images, not 2D feature vectors
- Use **Tab 3 (Loss Analysis)** → Latent Space scatter instead
        """)
        return   # FIX-2: EXIT EARLY — no crash from get_loss() or landscape compute

    st.markdown("""
    <div class="info-box">
    The <b>Loss Landscape</b> is computed by perturbing two weight directions
    orthogonally from the trained solution and evaluating the loss at each point.
    Flat regions = easy to optimise. Sharp minima = poor generalisation.
    Saddle points = training instability.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 3])
    with col_ctrl:
        grid_size = st.selectbox("Grid resolution (N×N)", [15, 20, 30], index=1,
                                 help="Higher = slower but smoother surface")
        alpha_range = st.slider("Perturbation range (α/β)", 0.1, 3.0, 1.0, step=0.1)

    with col_viz:
        with st.spinner("🔧 Computing loss landscape…"):
            from plots.loss_landscape import plot_loss_landscape
            from losses.loss_registry import get_loss

            exp    = EXPERIMENTS[ss.experiment_id]
            task   = exp["task_type"]

            # Grab test data for landscape computation
            data   = ss.dataset
            if task == "multiclass":
                X_lnd = data["X_test_flat"] if ss.model_type == "MLP" else data["X_test"]
                y_lnd = data["y_test_cat"]
            elif task == "binary":
                X_lnd = data["X_test"]
                y_lnd = data["y_test"]
            elif task == "regression":
                X_lnd = data["X_test"]
                y_lnd = data["y_test"]
            elif task == "reconstruction":
                X_lnd = data["X_test"]
                y_lnd = data["X_test"]
            else:
                X_lnd = data["X_test"]
                y_lnd = data.get("y_test", data["X_test"])

            # FIX-2: get_loss() now normalises loss_name to lowercase before
            # lookup, so display names like "Cross Entropy" work correctly.
            loss_fn = get_loss(
                ss.loss_name,
                huber_delta=ss.huber_delta,
                focal_alpha=ss.focal_alpha,
                focal_gamma=ss.focal_gamma,
            )

            fig_landscape = plot_loss_landscape(
                model     = ss.model,
                X         = X_lnd,
                y         = y_lnd,
                loss_fn   = loss_fn,
                grid_size = grid_size,
                title     = f"Loss Landscape — {ss.loss_name}",
            )

        # FIX-1 + FIX-4: unique key using experiment_id from session state
        # (no undefined variable — ss.get() is always safe)
        _exp_num = ss.get("experiment_id", 1)
        st.plotly_chart(
            fig_landscape,
            use_container_width=True,
            key=f"tab4_landscape_exp{_exp_num}_loss_surface_1",
        )

    st.caption(
        "🌋 **Interpretation key**: Blue valleys = low loss (good). Red peaks = high loss (bad). "
        "Yellow path = simulated optimizer trajectory. A wide, smooth valley → better generalisation."
    )


# ── TAB 5: Compare Mode ───────────────────────────────────────────────────────
def tab_compare_mode():
    ss = st.session_state

    if not ss.comparison_mode:
        st.info(
            "Enable **⚖️ Comparison Mode** in the sidebar to compare two loss functions side-by-side.",
            icon="ℹ️",
        )
        return

    if not ss.trained or ss.history_b is None:
        st.info("Train the model with Comparison Mode enabled.", icon="⏳")
        return

    from plots.loss_curves import plot_comparison_curves
    from plots.confusion_matrix import plot_confusion_matrix

    loss_a = ss.loss_name
    loss_b = ss.loss_name_b if ss.loss_name_b else "Loss B"
    exp    = EXPERIMENTS[ss.experiment_id]
    task   = exp["task_type"]

    st.markdown(f"### ⚖️ {loss_a} vs {loss_b}")

    # ── Training curves comparison ─────────────────────────────────────────
    fig_comp = plot_comparison_curves(
        ss.history, ss.history_b,
        label_a=loss_a, label_b=loss_b,
        task_type=task,
    )
    st.plotly_chart(fig_comp, use_container_width=True, key="compare_training_curves_1")

    # ── Metrics comparison table ───────────────────────────────────────────
    st.markdown("### 📊 Final Metrics Comparison")
    col1, col2 = st.columns(2)

    def _determine_winner(m_a, m_b, key, higher_better=True):
        va, vb = m_a.get(key), m_b.get(key)
        if va is None or vb is None:
            return None, None
        if higher_better:
            return ("A", "B") if va > vb else ("B", "A")
        else:
            return ("A", "B") if va < vb else ("B", "A")

    with col1:
        st.markdown(f"**{loss_a}**")
        render_metric_cards(ss.metrics)

    with col2:
        st.markdown(f"**{loss_b}**")
        if ss.metrics_b:
            render_metric_cards(ss.metrics_b)

    # Winner badge
    st.divider()
    if ss.metrics and ss.metrics_b:
        # Use F1 for classification, R² for regression
        if task in ("multiclass", "binary"):
            key = "F1 (weighted)" if "F1 (weighted)" in ss.metrics else "F1 Score"
            higher = True
        elif task == "regression":
            key = "R²"
            higher = True
        else:
            key = "MSE (pixel)"
            higher = False

        wa, wb = _determine_winner(ss.metrics, ss.metrics_b, key, higher_better=higher)
        if wa == "A":
            winner_nm = loss_a
            winner_col = COLORS["accuracy"]
        elif wa == "B":
            winner_nm = loss_b
            winner_col = COLORS["primary"]
        else:
            winner_nm  = "Tie"
            winner_col = COLORS["val_accent"]

        v_a = ss.metrics.get(key, "N/A")
        v_b = ss.metrics_b.get(key, "N/A") if ss.metrics_b else "N/A"
        st.markdown(f"""
        <div style="text-align:center; padding:16px;">
            <div style="font-size:0.9rem; color:#A0A8C0; margin-bottom:8px;">
                {key}: {loss_a}={v_a} vs {loss_b}={v_b}
            </div>
            <div style="display:inline-block; background:linear-gradient(135deg,
                {winner_col}55,{winner_col}22);
                border:2px solid {winner_col}; border-radius:12px;
                padding:10px 24px; font-size:1.2rem; font-weight:700; color:{winner_col};">
                🏆 Winner: {winner_nm}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Side-by-side confusion matrices (if classification) ────────────────
    if task in ("multiclass", "binary") and ss.y_pred_b is not None:
        st.divider()
        st.markdown("### 🎯 Confusion Matrices")
        col1, col2 = st.columns(2)

        class_names = (
            [str(i) for i in range(10)] if task == "multiclass"
            else ["Class 0", "Class 1"]
        )
        with col1:
            fig_cm_a = plot_confusion_matrix(
                ss.y_true_test, ss.y_pred, class_names=class_names,
                title=f"{loss_a}", threshold=ss.threshold,
            )
            st.plotly_chart(fig_cm_a, use_container_width=True, key="compare_confusion_matrix_a_1")
        with col2:
            fig_cm_b = plot_confusion_matrix(
                ss.y_true_test, ss.y_pred_b, class_names=class_names,
                title=f"{loss_b}", threshold=ss.threshold,
            )
            st.plotly_chart(fig_cm_b, use_container_width=True, key="compare_confusion_matrix_b_1")


# ── TAB 6: Explainer ──────────────────────────────────────────────────────────
def tab_explainer():
    ss = st.session_state
    from utils.explainer import generate_explainer_content

    loss_nm = ss.loss_name
    exp_id  = ss.experiment_id
    exp     = EXPERIMENTS[exp_id]

    content = generate_explainer_content(
        loss_name     = loss_nm,
        experiment_id = exp_id,
        task_type     = exp["task_type"],
    )

    st.markdown(f"## 💡 Understanding: **{loss_nm}**")

    # Formula
    st.markdown("### 📐 Mathematical Formula")
    st.latex(content["latex"])

    col1, col2 = st.columns([3, 2])

    with col1:
        # Description
        st.markdown("### 📖 What Does It Measure?")
        st.markdown(content["description"])

        # Experiment context
        if content["experiment_context"]:
            st.markdown("### 🔬 In This Experiment")
            st.markdown(
                f'<div class="info-box">{content["experiment_context"]}</div>',
                unsafe_allow_html=True,
            )

        # Wrong loss consequence
        if content["wrong_consequence"]:
            st.markdown("### ⚠️ What If You Use the Wrong Loss?")
            st.markdown(
                f'<div class="warn-box">{content["wrong_consequence"]}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        # When to use
        st.markdown("### ✅ When to Use")
        st.markdown(content["when_to_use"])

        st.markdown("### ❌ When NOT to Use")
        st.markdown(content["when_not_to_use"])

        # Viva tip
        if content["viva_tip"]:
            st.markdown(
                f'<div class="tip-box">{content["viva_tip"]}</div>',
                unsafe_allow_html=True,
            )

    # Comparison table
    if content["comparison"]:
        st.divider()
        st.markdown("### 📊 Comparison with Alternatives")
        st.markdown(content["comparison"])

    # All loss functions summary at bottom
    st.divider()
    st.markdown("### 🗂️ Quick Reference: All Loss Functions")
    summary_rows = []
    for nm, meta in LOSS_METADATA.items():
        summary_rows.append({
            "Loss Function":  nm,
            "Category":       meta["category"].replace("_", " ").title(),
            "Keras Class":    meta["keras_name"],
            "Use Case":       meta["when_to_use"].split("\n")[0].strip("- "),
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    train_clicked, status_placeholder = render_sidebar()

    render_header()

    # Trigger training
    if train_clicked:
        run_training(status_placeholder)

    # Main tabs
    tab_labels = [
        "📊 Training Dashboard",
        "🔍 Predictions",
        "📉 Loss Analysis",
        "🌋 Loss Landscape",
        "⚖️ Compare Mode",
        "💡 Explainer",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_training_dashboard()

    with tabs[1]:
        tab_predictions()

    with tabs[2]:
        tab_loss_analysis()

    with tabs[3]:
        tab_loss_landscape()

    with tabs[4]:
        tab_compare_mode()

    with tabs[5]:
        tab_explainer()


if __name__ == "__main__":
    main()
