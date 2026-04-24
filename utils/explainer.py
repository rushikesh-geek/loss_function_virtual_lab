# =============================================================================
# utils/explainer.py
# Auto-generates rich educational content for the selected loss function.
# Draws from config.LOSS_METADATA to produce Streamlit-renderable content.
# =============================================================================

from config import LOSS_METADATA, EXPERIMENTS


def generate_explainer_content(
    loss_name: str,
    experiment_id: int = None,
    task_type: str = None,
) -> dict:
    """
    Generate comprehensive educational content for a given loss function.

    Parameters
    ----------
    loss_name     : friendly loss name (must match key in LOSS_METADATA)
    experiment_id : current experiment number (for context-specific tips)
    task_type     : "multiclass" | "binary" | "regression" | "reconstruction"

    Returns
    -------
    dict with keys:
        latex       : LaTeX string for st.latex()
        description : markdown-formatted description
        when_to_use : markdown list
        when_not_to_use: markdown list
        viva_tip    : markdown string with exam tip
        wrong_consequence : what happens if you use wrong loss
        experiment_context: experiment-specific insight (if applicable)
        comparison  : comparison with alternative losses
    """
    meta = LOSS_METADATA.get(loss_name)
    if meta is None:
        return {
            "latex":               r"L = \text{Unknown Loss}",
            "description":         f"No metadata found for loss: **{loss_name}**",
            "when_to_use":         "",
            "when_not_to_use":     "",
            "viva_tip":            "",
            "wrong_consequence":   "",
            "experiment_context":  "",
            "comparison":          "",
        }

    # ── Experiment context (why this loss for this experiment) ────────────────
    experiment_context = _get_experiment_context(loss_name, experiment_id)

    # ── Comparison with alternatives ──────────────────────────────────────────
    comparison = _get_comparison(loss_name)

    return {
        "latex":               meta["latex"],
        "description":         meta["description"],
        "when_to_use":         meta["when_to_use"],
        "when_not_to_use":     meta["when_not_to_use"],
        "viva_tip":            meta["viva_tip"],
        "wrong_consequence":   meta["wrong_loss_consequence"],
        "experiment_context":  experiment_context,
        "comparison":          comparison,
    }


def _get_experiment_context(loss_name: str, experiment_id: int) -> str:
    """Return a specific insight string for this loss in the current experiment."""
    contexts = {
        ("Cross Entropy", 1): (
            "**In Experiment 1**, Cross-Entropy directly maximises the log-probability "
            "of the correct digit class. With softmax, the gradient signal is proportional "
            "to the prediction error (p̂ - y), which is always non-zero when wrong — "
            "unlike MSE whose sigmoid derivative saturates."
        ),
        ("MSE", 1): (
            "**In Experiment 1**, using MSE on MNIST treats digit labels as ordinal numbers "
            "(Label 9 is 'between' 8 and 10). The network has no incentive to produce "
            "a proper probability distribution, and training is much slower due to gradient saturation."
        ),
        ("Binary Cross Entropy", 2): (
            "**In Experiment 2**, BCE is ideal: each patient is either malignant or benign. "
            "The threshold slider lets you explore the precision-recall tradeoff. "
            "Lowering the threshold catches more cancer (higher recall) at the cost of "
            "more false positives (lower precision) — a critical clinical decision."
        ),
        ("MSE", 3): (
            "**In Experiment 3**, MSE is sensitive to the injected outliers. "
            "Each outlier contributes quadratically to the gradient: if a price prediction "
            "is 10× off, it contributes 100× to the loss vs a 1× error. "
            "The network spends capacity fitting outliers instead of the main distribution."
        ),
        ("Huber", 3): (
            "**In Experiment 3**, Huber loss caps the gradient for large residuals at δ. "
            "Even with 200 outliers, the gradient from each outlier is at most δ — "
            "preventing them from dominating the weight update. Increase δ for more "
            "MSE-like behaviour, decrease for more MAE-like (more robust)."
        ),
        ("Binary Cross Entropy", 4): (
            "**In Experiment 4**, BCE suffers from the accuracy/F1 paradox: "
            "at 1% fraud rate, predicting ALL NORMAL gives 99% accuracy but F1≈0. "
            "The loss treats each sample equally — 99 normal samples overwhelm "
            "the gradient from 1 fraud sample. Switch to Focal Loss to fix this."
        ),
        ("Focal Loss", 4): (
            "**In Experiment 4**, Focal Loss rescues minority class detection by "
            "down-weighting the 99% easy normal samples. With γ=2, each normal sample "
            "confident at p=0.95 only contributes (0.05)²=0.0025 of its original BCE loss "
            "while the fraud samples dominate training. Increase γ to focus even harder."
        ),
        ("MSE", 5): (
            "**In Experiment 5 (MSE Reconstruction)**, MSE computes pixel-wise reconstruction error. "
            "The autoencoder's bottleneck must compress each digit into just latent_dim numbers, then "
            "reconstruct all 784 pixels. MSE assumes Gaussian pixel noise — producing smooth but "
            "slightly blurry digits (the model hedges by outputting the mean of all plausible completions)."
        ),
        # Experiment 5 — new loss modes
        ("MSE Reconstruction", 5): (
            "**In Experiment 5 (MSE Reconstruction)**, MSE is the simplest reconstruction baseline. "
            "The autoencoder minimises average squared pixel error between input and output. "
            "Works well but produces blurry results because the model averages over all possible "
            "digit completions when the bottleneck is small. Compare with BCE for sharper outputs."
        ),
        ("BCE Reconstruction", 5): (
            "**In Experiment 5 (BCE Reconstruction)**, each pixel is treated as a Bernoulli variable. "
            "BCE = -Σ[x·log(x̂)+(1-x)·log(1-x̂)] requires sigmoid output. "
            "For MNIST (near-binary pixels), BCE produces sharper, higher-contrast digits than MSE "
            "because it directly models the on/off nature of ink pixels. "
            "The sigmoid activation clips reconstructions to [0,1] matching the normalised input."
        ),
        ("VAE Loss (Recon + KL)", 5): (
            "**In Experiment 5 (VAE Loss)**, the ELBO = Recon + β·KL is maximised. "
            "The reconstruction term (BCE) encourages accurate pixel reproduction; "
            "the KL term −½Σ(1+log_var-μ²-exp(log_var)) penalises the encoder for straying "
            "from the prior N(0,I), creating a smooth, structured latent space. "
            "Notice 3 separate loss curves in Tab 1 and the KL-per-dim chart in Tab 3. "
            "Set β>1 (sidebar) for more disentangled representations (β-VAE); "
            "set latent_dim=2 to see the smooth, continuous 2D clusters."
        ),
        ("Denoising AE (MSE)", 5): (
            "**In Experiment 5 (Denoising AE)**, Gaussian noise (σ=noise_factor) is added to the input "
            "before the encoder. The target remains the CLEAN original. "
            "The model must extract digit structure and discard noise — forcing it to learn "
            "the manifold of valid digits rather than a trivial pass-through. "
            "Tab 2 shows all 3 columns: Noisy Input | Denoised Output | Clean Reference. "
            "Increase noise_factor in the sidebar to make the task harder."
        ),
    }
    key = (loss_name, experiment_id)
    return contexts.get(key, "")


def _get_comparison(loss_name: str) -> str:
    """Return a comparison table markdown string for the given loss."""
    comparisons = {
        "Cross Entropy": (
            "| Feature | Cross Entropy | MSE |\n"
            "|---------|--------------|-----|\n"
            "| Output activation | Softmax | Any |\n"
            "| Gradient at saturation | ✅ Non-zero | ❌ Near-zero |\n"
            "| Treats labels as | Probabilities | Numbers |\n"
            "| Convergence speed | Fast | Slow |\n"
            "| Use for classification | ✅ Always | ❌ Avoid |"
        ),
        "Binary Cross Entropy": (
            "| Feature | BCE | Focal Loss |\n"
            "|---------|-----|------------|\n"
            "| Balanced data | ✅ Works well | ✅ Works (γ=0=BCE) |\n"
            "| Imbalanced data | ❌ Accuracy paradox | ✅ Rescues minority |\n"
            "| Easy examples | Equal weight | Down-weighted |\n"
            "| Hyperparameters | None | α, γ |\n"
            "| Default choice | Yes | When imbalance > 10:1 |"
        ),
        "MSE": (
            "| Feature | MSE | MAE | Huber |\n"
            "|---------|-----|-----|-------|\n"
            "| Differentiable at 0 | ✅ | ❌ | ✅ |\n"
            "| Outlier robustness | ❌ Poor | ✅ Good | ✅ Good |\n"
            "| Gaussian noise | ✅ Optimal | Suboptimal | ✅ Good |\n"
            "| Gradient magnitude | Proportional to error | Constant | Mixed |\n"
            "| Use case | Clean data | Heavy tails | Occasional outliers |"
        ),
        "Huber": (
            "| Feature | Huber (δ=1) | MSE | MAE |\n"
            "|---------|-------------|-----|-----|\n"
            "| Small errors | MSE-like | MSE | MAE |\n"
            "| Large errors | MAE-like | ❌ Dominates | MAE |\n"
            "| Differentiable everywhere | ✅ | ✅ | ❌ (at 0) |\n"
            "| δ control | Yes | N/A | N/A |\n"
            "| Gradient bound | Bounded by δ | Unbounded | ±1 |"
        ),
        "Focal Loss": (
            "| Feature | Focal Loss | BCE | Weighted BCE |\n"
            "|---------|-----------|-----|---------------|\n"
            "| Handles imbalance | ✅ Automatic | ❌ | ✅ Manual weights |\n"
            "| Easy example handling | ✅ Down-weight | Equal | Equal |\n"
            "| Parameters | α, γ | None | class_weight dict |\n"
            "| Paper | RetinaNet (2017) | Classic | Classic |\n"
            "| When γ=0 | = Weighted BCE | = BCE | = Weighted BCE |"
        ),
        "MSE Reconstruction": (
            "| Property | MSE AE | BCE AE | VAE | Denoising AE |\n"
            "|----------|--------|--------|-----|-------------|\n"
            "| Generative? | No | No | ✅ Yes | No |\n"
            "| Latent smooth? | No | No | ✅ Yes (KL) | No |\n"
            "| Sharpness | Low (blurry) | Medium | Medium | High |\n"
            "| Training stable? | ✅ Yes | ✅ Yes | Moderate | ✅ Yes |\n"
            "| Extra input | Clean | Clean | Clean | Noisy |\n"
            "| Use case | Baseline | MNIST | Generation | Denoising |\n"
            "| Key hyperparameter | latent_dim | latent_dim | β, latent_dim | noise_factor |"
        ),
        "BCE Reconstruction": (
            "| Property | MSE AE | BCE AE | VAE | Denoising AE |\n"
            "|----------|--------|--------|-----|-------------|\n"
            "| Generative? | No | No | ✅ Yes | No |\n"
            "| Latent smooth? | No | No | ✅ Yes (KL) | No |\n"
            "| Sharpness | Low (blurry) | Medium | Medium | High |\n"
            "| Training stable? | ✅ Yes | ✅ Yes | Moderate | ✅ Yes |\n"
            "| Extra input | Clean | Clean | Clean | Noisy |\n"
            "| Use case | Baseline | MNIST | Generation | Denoising |\n"
            "| Key hyperparameter | latent_dim | latent_dim | β, latent_dim | noise_factor |"
        ),
        "VAE Loss (Recon + KL)": (
            "| Property | Standard AE | VAE |\n"
            "|----------|-------------|-----|\n"
            "| Latent space | Arbitrary | ≈ Gaussian (KL-regularised) |\n"
            "| Generation | ❌ No | ✅ Yes (sample z ~ N(0,I)) |\n"
            "| Interpolation | Gaps / artifacts | Smooth |\n"
            "| Training | Simple (one loss) | ELBO (two terms) |\n"
            "| Sharpness | Higher | Moderate |\n"
            "| Posterior collapse risk | None | Yes (use β<1) |\n"
            "| β effect | N/A | β>1 → disentangled, β<1 → sharper |"
        ),
        "Denoising AE (MSE)": (
            "| Property | Standard AE | Denoising AE |\n"
            "|----------|-------------|-------------|\n"
            "| Input | Clean image | Clean + Gaussian noise |\n"
            "| Target | Clean image | Clean image |\n"
            "| Feature robustness | Lower | Higher (noise-invariant) |\n"
            "| Risk of identity | Medium | Low (noise prevents it) |\n"
            "| Noise level control | N/A | noise_factor slider |\n"
            "| Best for | General compression | Pre-training, denoising |"
        ),
    }
    return comparisons.get(loss_name, "")
