# =============================================================================
# config.py — Central configuration for the Interactive Loss Function Virtual Lab
# All constants, color schemes, loss descriptions, and LaTeX formulas live here.
# Students: Browse this file to understand the taxonomy of loss functions.
# =============================================================================

# ---------------------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------------------
APP_TITLE = "🧪 Interactive Loss Function Virtual Lab"
APP_SUBTITLE = "University Deep Learning Teaching Tool"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# COLOR PALETTE  (Dark-friendly, used across all Plotly charts)
# ---------------------------------------------------------------------------
COLORS = {
    "primary":       "#4A90D9",   # Main blue
    "secondary":     "#7EC8E3",   # Light blue accent
    "loss_accent":   "#FF6B6B",   # Red — used for loss curves
    "val_accent":    "#FFD93D",   # Yellow — used for val curves
    "accuracy":      "#6BCB77",   # Green — used for accuracy
    "background":    "#0E1117",   # Streamlit dark background
    "card_bg":       "#1E2130",   # Metric card background
    "border":        "#2D3250",   # Border color
    "text":          "#FAFAFA",   # Primary text
    "subtext":       "#A0A8C0",   # Secondary text
    "correct":       "#6BCB77",   # Correct prediction highlight
    "wrong":         "#FF6B6B",   # Wrong prediction highlight
    "gradient_start":"#1A1A2E",
    "gradient_end":  "#16213E",
}

# Plotly color sequences for multi-line charts
PLOTLY_COLORS = [
    "#4A90D9", "#FF6B6B", "#6BCB77", "#FFD93D",
    "#C77DFF", "#FF9A3C", "#00B4D8", "#F72585",
]

# ---------------------------------------------------------------------------
# EXPERIMENT DEFINITIONS
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    1: {
        "name": "Multi-class Classification",
        "icon": "🔢",
        "dataset": "MNIST",
        "description": (
            "Compare Cross-Entropy vs MSE on digit classification. "
            "Understand why probabilistic losses dominate for classification tasks."
        ),
        "losses": ["Cross Entropy", "MSE"],
        "models": ["MLP", "Small CNN"],
        "task_type": "multiclass",
    },
    2: {
        "name": "Binary Classification",
        "icon": "⚕️",
        "dataset": "Breast Cancer (sklearn)",
        "description": (
            "Binary Cross-Entropy on medical diagnosis data. "
            "Explore how decision threshold affects precision/recall tradeoff."
        ),
        "losses": ["Binary Cross Entropy"],
        "models": ["MLP"],
        "task_type": "binary",
    },
    3: {
        "name": "Regression + Outlier Robustness",
        "icon": "🏠",
        "dataset": "California Housing (sklearn)",
        "description": (
            "MSE vs Huber loss with injected outliers. "
            "See how MSE breaks under heavy-tailed noise while Huber stays robust."
        ),
        "losses": ["MSE", "Huber"],
        "models": ["Dense NN"],
        "task_type": "regression",
    },
    4: {
        "name": "Imbalanced Classification",
        "icon": "⚖️",
        "dataset": "Synthetic Fraud Dataset",
        "description": (
            "BCE vs Focal Loss on imbalanced data. "
            "Reveal the accuracy/F1 paradox and how Focal Loss rescues minority detection."
        ),
        "losses": ["Binary Cross Entropy", "Focal Loss"],
        "models": ["MLP"],
        "task_type": "binary",
    },
    5: {
        "name": "Autoencoder Reconstruction",
        "icon": "🔁",
        "dataset": "MNIST (reconstruction)",
        "description": (
            "Compare four reconstruction loss modes on a Convolutional Autoencoder.\n"
            "MSE vs BCE vs VAE (Recon+KL) vs Denoising — each teaches a different concept "
            "about latent representations and reconstruction quality."
        ),
        "losses": [
            "MSE Reconstruction",
            "BCE Reconstruction",
            "VAE Loss (Recon + KL)",
            "Denoising AE (MSE)",
        ],
        "models": ["Autoencoder"],
        "task_type": "reconstruction",
    },
}

# ---------------------------------------------------------------------------
# LOSS FUNCTION REGISTRY METADATA
# ---------------------------------------------------------------------------
LOSS_METADATA = {
    "Cross Entropy": {
        "keras_name":  "categorical_crossentropy",
        "short_name":  "CE",
        "category":    "classification",
        "latex":       r"L = -\sum_{i} y_i \log(\hat{y}_i)",
        "description": (
            "**Categorical Cross-Entropy** measures the dissimilarity between the "
            "true probability distribution and the predicted distribution. It is the "
            "gold standard for multi-class classification because it directly optimizes "
            "the log-likelihood of the correct class, giving large gradients when the "
            "model is confidently wrong."
        ),
        "when_to_use": (
            "- Multi-class classification (≥3 classes)\n"
            "- When the model output is a softmax probability vector\n"
            "- Any problem where class membership is mutually exclusive"
        ),
        "when_not_to_use": (
            "- Regression problems\n"
            "- Multi-label classification (use Binary CE per label)\n"
            "- When class probabilities are severely imbalanced (use Focal Loss instead)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: Cross-entropy = negative log-likelihood under a categorical "
            "distribution. Minimising CE is equivalent to maximising the likelihood of "
            "the training data assuming IID samples from a categorical distribution. "
            "Always pair with softmax, never sigmoid, for multi-class."
        ),
        "wrong_loss_consequence": (
            "Using MSE for classification treats class labels as continuous quantities (e.g., "
            "class 3 is 'between' classes 2 and 4). The gradient vanishes when the sigmoid "
            "output is far from 0.5, causing extremely slow learning for confident wrong predictions."
        ),
    },
    "Binary Cross Entropy": {
        "keras_name":  "binary_crossentropy",
        "short_name":  "BCE",
        "category":    "binary_classification",
        "latex":       r"L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]",
        "description": (
            "**Binary Cross-Entropy** is the special case of CE for two-class problems. "
            "It penalises confident wrong predictions logarithmically, acting as an "
            "asymmetric bounding force on the sigmoid output."
        ),
        "when_to_use": (
            "- Binary classification (spam/not spam, malignant/benign)\n"
            "- Multi-label classification (apply BCE independently per label)\n"
            "- Any sigmoid-output network"
        ),
        "when_not_to_use": (
            "- Severely imbalanced datasets (minority class recall collapses — use Focal Loss)\n"
            "- Regression problems"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: BCE is the log-loss. It equals the KL divergence between the "
            "true Bernoulli distribution and the predicted distribution (up to a constant). "
            "A perfect model → loss → 0; random model → loss → ln(2) ≈ 0.693."
        ),
        "wrong_loss_consequence": (
            "On highly imbalanced data (e.g., 1% fraud), a model predicting ALL ZEROS "
            "achieves 99% accuracy with low BCE — yet detects zero frauds. "
            "This is the accuracy/F1 paradox. Switch to Focal Loss or use class weights."
        ),
    },
    "MSE": {
        "keras_name":  "mean_squared_error",
        "short_name":  "MSE",
        "category":    "regression",
        "latex":       r"L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        "description": (
            "**Mean Squared Error** is the expected squared Euclidean distance between "
            "predictions and targets. It assumes Gaussian noise in labels and corresponds "
            "to maximum likelihood under a Gaussian likelihood."
        ),
        "when_to_use": (
            "- Regression with Gaussian noise\n"
            "- Reconstruction loss in autoencoders\n"
            "- Smooth, outlier-free targets"
        ),
        "when_not_to_use": (
            "- Data with heavy-tailed noise or outliers (use Huber/MAE)\n"
            "- Classification targets (use Cross-Entropy)\n"
            "- Ordinal targets (MSE treats them as cardinal)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: MSE penalises large errors quadratically — "
            "one outlier 10× away contributes 100× more to the gradient than "
            "one point 1× away. This is why a single outlier can dominate training. "
            "The gradient ∂L/∂ŷ = -2(y - ŷ)/n is proportional to the residual."
        ),
        "wrong_loss_consequence": (
            "Using MSE for classification causes gradient saturation: when sigmoid(z) is "
            "near 0 or 1 (confident predictions), ∂MSE/∂z → 0 and learning stalls. "
            "Cross-Entropy avoids this because its gradient does NOT include the sigmoid derivative."
        ),
    },
    "Huber": {
        "keras_name":  "huber",
        "short_name":  "Huber",
        "category":    "robust_regression",
        "latex":       (
            r"L_\delta = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & |y-\hat{y}|\leq\delta \\"
            r" \delta|y-\hat{y}| - \frac{\delta^2}{2} & \text{otherwise}\end{cases}"
        ),
        "description": (
            "**Huber Loss** (smooth L1) combines the best of MSE (smooth gradient near zero) "
            "and MAE (linear, robust to outliers). The δ hyperparameter controls the "
            "transition point — errors smaller than δ are treated as MSE, larger as MAE."
        ),
        "when_to_use": (
            "- Regression with occasional outliers\n"
            "- Object detection (bounding box regression)\n"
            "- Financial forecasting with fat-tailed noise"
        ),
        "when_not_to_use": (
            "- Clean Gaussian data (MSE is simpler and equivalent)\n"
            "- When you need exact gradients at 0 (Huber has a kink at δ transition)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: Huber = MAE for |error| > δ, MSE for |error| ≤ δ. "
            "It is everywhere differentiable (unlike MAE), which makes it compatible "
            "with gradient descent. δ is a hyperparameter: larger δ → more like MSE, "
            "smaller δ → more like MAE."
        ),
        "wrong_loss_consequence": (
            "Using MSE on outlier-heavy data causes the optimizer to spend most of its "
            "capacity fitting the outliers (large squared residuals dominate gradients), "
            "leading to poor performance on the actual clean data distribution."
        ),
    },
    "Focal Loss": {
        "keras_name":  "focal_loss",
        "short_name":  "FL",
        "category":    "imbalanced_classification",
        "latex":       r"FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)",
        "description": (
            "**Focal Loss** (Lin et al., 2017) extends BCE by adding a modulating factor "
            "(1−pt)^γ that down-weights easy examples and focuses training on the hard "
            "minority class. α balances positive/negative frequency, γ adjusts the "
            "hard-example focus curve."
        ),
        "when_to_use": (
            "- Heavily imbalanced classification (fraud, rare events, one-stage detection)\n"
            "- Object detection (RetinaNet original paper)\n"
            "- Any problem where BCE converges to predicting the majority class"
        ),
        "when_not_to_use": (
            "- Balanced datasets (BCE is sufficient)\n"
            "- When γ=0 (degenerates to weighted BCE — just use BCE with class_weight)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: γ=0 → Focal Loss = BCE. γ=2 is the default from the paper. "
            "With γ=2, an easy example (pt=0.9) is down-weighted by (0.1)^2 = 0.01 "
            "vs a hard example (pt=0.1) which gets (0.9)^2 = 0.81 — 81× more attention. "
            "α compensates for class frequency imbalance."
        ),
        "wrong_loss_consequence": (
            "BCE on 1% fraud data: the model ignores minority class entirely. "
            "Accuracy looks great (99% correct by predicting all-negative), but F1 for "
            "fraud class is ≈0. Focal Loss forces the model to attend to the hard "
            "minority examples by suppressing the easy majority signal."
        ),
    },
    # ── Experiment 5 loss modes ────────────────────────────────────────────────
    "MSE Reconstruction": {
        "keras_name":  "mean_squared_error",
        "short_name":  "MSE-AE",
        "category":    "reconstruction",
        "latex":       r"L = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2",
        "description": (
            "**MSE Reconstruction** treats each pixel as a continuous Gaussian variable. "
            "The autoencoder minimises the average squared pixel-level error between "
            "input and reconstruction. Simple and stable, but assumes pixel noise is "
            "Gaussian — leading to slightly blurry outputs as the model hedges across modes."
        ),
        "when_to_use": (
            "- Autoencoder baseline / ablation study\n"
            "- Continuous-valued inputs (e.g., audio, depth maps)\n"
            "- When training stability matters more than sharpness"
        ),
        "when_not_to_use": (
            "- Near-binary images (MNIST): BCE gives sharper results\n"
            "- When you need a generative model (use VAE)\n"
            "- When noise robustness matters (use Denoising AE)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: MSE assumes Gaussian pixel noise → MLE under N(x, σ²I). "
            "The blurriness of MSE reconstructions is a consequence of this assumption: "
            "the model hedges by predicting the mean of possible completions when "
            "the bottleneck is too small to uniquely identify the digit."
        ),
        "wrong_loss_consequence": (
            "MSE minimisation encourages the decoder to output the mean over all "
            "plausible reconstructions — leading to blurry images. BCE avoids this "
            "by treating each pixel as a Bernoulli trial, producing sharper outputs."
        ),
    },
    "BCE Reconstruction": {
        "keras_name":  "binary_crossentropy",
        "short_name":  "BCE-AE",
        "category":    "reconstruction",
        "latex":       r"L = -\sum_i \left[x_i \log\hat{x}_i + (1-x_i)\log(1-\hat{x}_i)\right]",
        "description": (
            "**BCE Reconstruction** treats each pixel as a Bernoulli probability. "
            "The decoder outputs x̂ᵢ ∈ (0,1) via sigmoid, interpreted as the probability "
            "that pixel i is 'on'. This is well-matched to MNIST digits (pixels are "
            "nearly binary), producing sharper reconstructions than MSE. "
            "Requires sigmoid output activation and [0,1]-normalised input."
        ),
        "when_to_use": (
            "- Near-binary images (MNIST, binary masks, scanned documents)\n"
            "- When you want sharper, higher-contrast reconstructions\n"
            "- VAE reconstruction term (standard choice in VAE papers)"
        ),
        "when_not_to_use": (
            "- Continuous-valued images where pixels are NOT near 0 or 1\n"
            "- Without sigmoid output activation (BCE requires ŷ ∈ (0,1))\n"
            "- Regression-style reconstruction of real-valued signals"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: BCE reconstruction = MLE under a Bernoulli pixel model. "
            "It is the default reconstruction loss in the original VAE paper (Kingma & Welling, 2013). "
            "BCE requires the final decoder activation to be sigmoid — using linear output with "
            "BCE produces undefined log(ŷ) for ŷ < 0."
        ),
        "wrong_loss_consequence": (
            "Using BCE with a linear (no sigmoid) decoder output causes log(ŷ) to blow up "
            "for negative ŷ values, producing NaN losses. Always pair BCE with a sigmoid "
            "output activation for autoencoders."
        ),
    },
    "VAE Loss (Recon + KL)": {
        "keras_name":  "elbo",
        "short_name":  "VAE",
        "category":    "generative",
        "latex":       (
            r"\mathcal{L}_{\text{VAE}} = "
            r"\underbrace{\mathbb{E}_{q}[\log p(x|z)]}_{\text{Reconstruction}}"
            r" - \beta \cdot "
            r"\underbrace{D_{\text{KL}}(q(z|x)\|p(z))}_{\text{KL Divergence}}"
        ),
        "description": (
            "**VAE Loss** = ELBO (Evidence Lower Bound) = reconstruction loss + β·KL divergence.\n\n"
            "• **Reconstruction term**: BCE(x, x̂) — how well the image is reproduced\n"
            "• **KL term**: D_KL(q(z|x)‖p(z)) = -½Σ(1 + log_var - μ² - exp(log_var)) — "
            "penalises the encoder for straying from the standard Gaussian prior N(0,I)\n"
            "• **β**: scaling factor (β=1 → original VAE; β>1 → β-VAE with more disentanglement)\n\n"
            "The KL term regularises the latent space, making it continuous and smooth, which "
            "enables generation by sampling z ~ N(0,I) directly (no encoder needed at inference)."
        ),
        "when_to_use": (
            "- Generative models (you want to SAMPLE new data, not just reconstruct)\n"
            "- When a smooth, interpolatable latent space is needed\n"
            "- Anomaly detection (OOD samples have high reconstruction loss + high KL)\n"
            "- Representation learning with structured, disentangled factors"
        ),
        "when_not_to_use": (
            "- When you only need compression/denoising (standard AE is simpler)\n"
            "- When training stability is critical (KL term can cause posterior collapse)\n"
            "- Very small datasets (VAE needs enough data to estimate posterior)"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: The VAE ELBO = log p(x) − D_KL(q(z|x)‖p(z|x)). "
            "Maximising the ELBO is equivalent to maximising a lower bound on the "
            "log-likelihood log p(x) while regularising the posterior. "
            "The reparameterisation trick z = μ + ε·σ (ε~N(0,1)) makes this differentiable. "
            "KL collapse (KL→0 early) is a known failure mode — use β<1 to prevent it."
        ),
        "wrong_loss_consequence": (
            "Training a VAE with loss= in model.compile() when a custom train_step is defined "
            "will cause the compiled loss to conflict with the ELBO in train_step. "
            "Always compile a VAE with ONLY an optimizer. "
            "Using too high β causes KL collapse — all latent dims become N(0,1) and the "
            "decoder ignores the latent code entirely (posterior collapse)."
        ),
    },
    "Denoising AE (MSE)": {
        "keras_name":  "mean_squared_error",
        "short_name":  "DAE",
        "category":    "reconstruction",
        "latex":       r"L = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_{i,\text{noisy}})^2",
        "description": (
            "**Denoising Autoencoder** adds Gaussian noise to the input and trains to "
            "reconstruct the CLEAN original: L = MSE(x_clean, Decoder(Encoder(x_noisy))).\n\n"
            "This denoising objective prevents the model from learning the identity function. "
            "The encoder must extract noise-invariant features (the 'signal') and ignore "
            "the random noise — producing more robust, generalisable representations "
            "than a standard AE trained on clean data."
        ),
        "when_to_use": (
            "- Data augmentation via corruption (image denoising, speech enhancement)\n"
            "- Robust feature learning pre-training\n"
            "- When clean labels are unavailable but noisy data is plentiful\n"
            "- Understanding what structure the AE has learned (test on progressively noisier inputs)"
        ),
        "when_not_to_use": (
            "- When your data is already noisy and you don't have clean targets\n"
            "- When you need a generative model (use VAE instead)\n"
            "- Very high noise factor > 0.6: the task becomes too hard for the bottleneck"
        ),
        "viva_tip": (
            "🎓 **Exam Tip**: Vincent et al. (2008) showed that a denoising AE trained to "
            "reconstruct clean x from noisy x̃ is equivalent to learning the score function "
            "∇_x log p(x) — the gradient of the data density. This connects denoising AEs "
            "to score-based generative models and diffusion models."
        ),
        "wrong_loss_consequence": (
            "If you use the noisy input as BOTH input and target (target=noisy instead of "
            "target=clean), the model learns to be a pass-through for noise rather than "
            "a denoiser. Always pass the clean version as y_train in model.fit() "
            "and the noisy version as X_train."
        ),
    },
}

# ---------------------------------------------------------------------------
# OPTIMIZER CONFIGS
# ---------------------------------------------------------------------------
OPTIMIZERS = {
    "Adam":    {"tensorflow_name": "adam",    "default_lr": 1e-3},
    "SGD":     {"tensorflow_name": "sgd",     "default_lr": 1e-2},
    "RMSprop": {"tensorflow_name": "rmsprop", "default_lr": 1e-3},
}

# ---------------------------------------------------------------------------
# DEFAULT HYPERPARAMETERS
# ---------------------------------------------------------------------------
DEFAULTS = {
    "epochs":      10,
    "batch_size":  64,
    "learning_rate": 1e-3,
    "dropout":     0.2,
    "optimizer":   "Adam",
    "huber_delta": 1.0,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "threshold":   0.5,
    "n_outliers":  50,
    "outlier_magnitude": 5.0,
    "fraud_ratio": 0.05,
    # Experiment 5 (Autoencoder)
    "latent_dim":  16,      # bottleneck dimension (set to 2 for 2D scatter)
    "vae_beta":    1.0,     # β-VAE weight on KL term
    "noise_factor": 0.3,   # Gaussian noise std for Denoising AE
}

# ---------------------------------------------------------------------------
# CHART CONFIG
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 400
LANDSCAPE_GRID = 30   # N×N grid for loss landscape computation

# ---------------------------------------------------------------------------
# CSS INJECTION
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Global font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%);
    border-right: 1px solid #2D3250;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1E2130 0%, #252B40 100%);
    border: 1px solid #2D3250;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(74,144,217,0.2);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4A90D9;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.78rem;
    color: #A0A8C0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Winner Badge ── */
.winner-badge {
    background: linear-gradient(135deg, #6BCB77, #4A9F52);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7EC8E3;
    border-bottom: 2px solid #2D3250;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ── Experiment Badge ── */
.exp-badge {
    background: linear-gradient(135deg, #1E3A5F, #2D5F8A);
    border: 1px solid #4A90D9;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    color: #7EC8E3;
    font-size: 0.9rem;
}

/* ── Info Box ── */
.info-box {
    background: rgba(74,144,217,0.08);
    border-left: 4px solid #4A90D9;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #FAFAFA;
}

/* ── Warn Box ── */
.warn-box {
    background: rgba(255,107,107,0.08);
    border-left: 4px solid #FF6B6B;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #FAFAFA;
}

/* ── Tip Box ── */
.tip-box {
    background: rgba(107,203,119,0.08);
    border-left: 4px solid #6BCB77;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #FAFAFA;
}

/* ── Strealit override: tabs ── */
[data-baseweb="tab-list"] {
    background: #1E2130;
    border-radius: 8px;
    padding: 4px;
}

[data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 500;
}

/* ── Progress hide default ugly bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4A90D9, #7EC8E3);
    border-radius: 4px;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #2D3250;
    margin: 20px 0;
}
</style>
"""
