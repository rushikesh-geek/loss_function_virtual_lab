# =============================================================================
# models/autoencoder.py
# Autoencoder family for Experiment 5 — MNIST Reconstruction
#
# Four distinct architectures, each teaching a different concept:
#
#  build_autoencoder()     → Standard Conv AE (MSE or BCE loss)
#  build_vae()             → Variational Autoencoder (VAE) + Sampling layer
#  VAE                     → tf.keras.Model subclass with custom train/test step
#  build_denoising_ae()    → Denoising AE (same arch, noisy input → clean target)
#
# ARCHITECTURE REFERENCE:
#   ENCODER: Input(28,28,1) → Conv2D(32) → Conv2D(64,stride=2) → Flatten → Dense
#   DECODER: Dense → Reshape(7,7,64) → ConvTranspose(64) → ConvTranspose(32,stride=2)
#            → ConvTranspose(1,sigmoid)
#   WHY CONV? Spatial structure of images: nearby pixels have high correlation.
#             Conv layers exploit this; Dense layers cannot.
#   WHY SIGMOID? Final activation maps pixel values to [0,1], matching normalised input.
# =============================================================================

import tensorflow as tf
from tensorflow import keras
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. STANDARD AUTOENCODER  (MSE or BCE reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(
    input_shape: tuple = (28, 28, 1),
    latent_dim: int    = 16,          # default 16; set to 2 for 2D scatter plot
    learning_rate: float = 1e-3,
    optimizer_name: str  = "Adam",
    loss_name: str       = "MSE",    # "MSE" | "BCE"
) -> tuple:
    """
    Build a Convolutional Autoencoder for MNIST reconstruction.

    Supports two reconstruction losses:
      • MSE  — pixel-wise mean squared error. Assumes Gaussian pixel noise.
               Produces smooth but slightly blurry reconstructions.
      • BCE  — binary cross-entropy per pixel. Treats each pixel as a
               Bernoulli probability. Sharper output; needs sigmoid final layer.
               Better for near-binary inputs like MNIST (mostly 0s and 1s).

    Architecture
    ────────────
    ENCODER:
        Input (28,28,1)
        → Conv2D(32, 3×3, relu, padding=same) → (28,28,32)
        → Conv2D(64, 3×3, relu, stride=2, padding=same) → (14,14,64)
        → Flatten → Dense(256, relu) → Dense(latent_dim)  ← BOTTLENECK

    DECODER:
        Dense(7×7×64, relu) → Reshape(7,7,64)
        → Conv2DTranspose(64, 3×3, relu, padding=same) → (7,7,64)
        → Conv2DTranspose(32, 3×3, relu, stride=2, padding=same) → (14,14,32)
        → Conv2DTranspose(16, 3×3, relu, stride=2, padding=same) → (28,28,16)
        → Conv2DTranspose(1, 3×3, sigmoid, padding=same)         → (28,28,1)

    Parameters
    ----------
    input_shape    : (H, W, C) — (28, 28, 1) for MNIST
    latent_dim     : bottleneck dimension (2 for 2D scatter plot)
    learning_rate  : optimizer learning rate
    optimizer_name : "Adam" | "SGD" | "RMSprop"
    loss_name      : "MSE" | "BCE" — which reconstruction loss to compile with

    Returns
    -------
    (autoencoder, encoder, decoder) — all three Keras models
    """
    optimizer = _build_optimizer(optimizer_name, learning_rate)

    # ── ENCODER ──────────────────────────────────────────────────────────────
    enc_input = keras.Input(shape=input_shape, name="enc_input")

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                             name="enc_conv1")(enc_input)       # (28,28,32)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", strides=2,
                             padding="same", name="enc_conv2")(x)  # (14,14,64)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                             name="enc_conv3")(x)               # (14,14,64)
    x = keras.layers.Flatten(name="enc_flat")(x)
    x = keras.layers.Dense(128, activation="relu", name="enc_dense1")(x)
    latent = keras.layers.Dense(latent_dim, name="latent")(x)   # BOTTLENECK

    encoder = keras.Model(enc_input, latent, name="Encoder")

    # ── DECODER ──────────────────────────────────────────────────────────────
    dec_input = keras.Input(shape=(latent_dim,), name="dec_input")

    x = keras.layers.Dense(7 * 7 * 64, activation="relu", name="dec_dense1")(dec_input)
    x = keras.layers.Reshape((7, 7, 64), name="dec_reshape")(x)

    x = keras.layers.Conv2DTranspose(64, (3, 3), activation="relu",
                                      padding="same", name="dec_convT1")(x)  # (7,7,64)
    x = keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", strides=2,
                                      padding="same", name="dec_convT2")(x)  # (14,14,32)
    x = keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", strides=2,
                                      padding="same", name="dec_convT3")(x)  # (28,28,16)
    # Sigmoid final layer: maps any value → [0,1] to match normalised input.
    # Required for BCE. Also works for MSE (slightly constrains output range).
    reconstructed = keras.layers.Conv2DTranspose(1, (3, 3), activation="sigmoid",
                                                  padding="same", name="output")(x)

    decoder = keras.Model(dec_input, reconstructed, name="Decoder")

    # ── FULL AUTOENCODER (Encoder → Decoder) ─────────────────────────────────
    ae_output   = decoder(encoder(enc_input))
    autoencoder = keras.Model(enc_input, ae_output, name="Autoencoder")

    # ── Compile with appropriate loss ─────────────────────────────────────────
    # BCE interpretation: each pixel xᵢ ∈ [0,1] is a Bernoulli probability;
    # the decoder output x̂ᵢ is the predicted probability.  BCE = -Σ[x·log(x̂)+(1-x)log(1-x̂)].
    # This tends to produce sharper digits than MSE for MNIST.
    if loss_name.upper() in ("BCE", "BINARY CROSS ENTROPY", "BINARY_CROSS_ENTROPY"):
        compiled_loss = "binary_crossentropy"
    else:
        compiled_loss = "mse"   # default: pixel-wise MSE

    autoencoder.compile(
        optimizer = optimizer,
        loss      = compiled_loss,
        metrics   = ["mae"],
    )

    return autoencoder, encoder, decoder


# ─────────────────────────────────────────────────────────────────────────────
# 2. VARIATIONAL AUTOENCODER  (VAE Loss: Reconstruction + KL Divergence)
# ─────────────────────────────────────────────────────────────────────────────

class Sampling(keras.layers.Layer):
    """
    Reparameterisation Trick sampling layer.

    WHY REPARAMETERISATION?
    -----------------------
    We want to sample z ~ q(z|x) = N(μ, σ²) and backpropagate through the
    sample.  Direct sampling is not differentiable.

    Trick: rewrite z = μ + ε·σ where ε ~ N(0,1) is sampled independently.
    Now ∂z/∂μ = 1 and ∂z/∂σ = ε — both differentiable.
    This allows gradients to flow back to the encoder's μ and log_var outputs.

    Input:  [z_mean, z_log_var]  — both of shape (batch, latent_dim)
    Output: z (sampled)           — shape (batch, latent_dim)
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch   = tf.shape(z_mean)[0]
        dim     = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        # σ = exp(0.5 · log_var); z = μ + ε·σ
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(
    input_shape: tuple = (28, 28, 1),
    latent_dim: int    = 8,
) -> tuple:
    """
    Build encoder and decoder sub-models for the VAE.

    The encoder outputs THREE tensors:
        z_mean    — mean of the approximate posterior q(z|x)
        z_log_var — log-variance of q(z|x)
        z         — one sample via the reparameterisation trick

    The decoder takes a latent sample z and reconstructs x.

    Parameters
    ----------
    input_shape : (H, W, C) for input images
    latent_dim  : dimension of the latent Gaussian space

    Returns
    -------
    (encoder, decoder) — used to instantiate VAE(encoder, decoder)

    NOTE: Do NOT compile the VAE with a loss= argument.
    The VAE class implements its own train_step and test_step that compute
    the ELBO = reconstruction loss + β·KL divergence.
    Passing loss= would break this.  Use: vae.compile(optimizer=Adam())
    """
    # ── Encoder ───────────────────────────────────────────────────────────────
    enc_inputs = keras.Input(shape=input_shape, name="enc_input")

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                             name="enc_conv1")(enc_inputs)         # (28,28,32)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", strides=2,
                             padding="same", name="enc_conv2")(x)  # (14,14,64)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                             name="enc_conv3")(x)                  # (14,14,64)
    x = keras.layers.Flatten(name="enc_flat")(x)
    x = keras.layers.Dense(128, activation="relu", name="enc_dense")(x)

    # Two separate Dense heads: mean and log-variance
    # log_var instead of σ directly prevents numerical issues (log_var ∈ ℝ freely)
    z_mean    = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    # Sample from q(z|x) using reparameterisation
    z         = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(enc_inputs, [z_mean, z_log_var, z], name="VAE_Encoder")

    # ── Decoder ───────────────────────────────────────────────────────────────
    # Maps a latent vector z back to an image.  Mirrors the encoder architecture.
    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")

    x = keras.layers.Dense(7 * 7 * 64, activation="relu", name="dec_dense")(latent_inputs)
    x = keras.layers.Reshape((7, 7, 64), name="dec_reshape")(x)
    x = keras.layers.Conv2DTranspose(64, (3, 3), activation="relu",
                                      padding="same", name="dec_convT1")(x)   # (7,7,64)
    x = keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", strides=2,
                                      padding="same", name="dec_convT2")(x)   # (14,14,32)
    x = keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", strides=2,
                                      padding="same", name="dec_convT3")(x)   # (28,28,16)
    # Sigmoid: pixels ∈ [0,1], matches normalised MNIST input
    outputs = keras.layers.Conv2DTranspose(1, (3, 3), activation="sigmoid",
                                            padding="same", name="dec_output")(x)

    decoder = keras.Model(latent_inputs, outputs, name="VAE_Decoder")

    return encoder, decoder


class VAE(keras.Model):
    """
    Variational Autoencoder — custom Keras model with built-in ELBO loss.

    Loss: L_VAE = E_q[log p(x|z)] − β · D_KL(q(z|x) ‖ p(z))
                = reconstruction_loss + β · kl_loss

    Where:
      • reconstruction_loss = BCE(x, x̂)  summed over pixels, averaged over batch
      • kl_loss             = -½ · Σ(1 + log_var - μ² - exp(log_var))  per latent dim
      • β                   = β-VAE scaling factor (β=1 → original VAE; β>1 → disentangled)

    IMPORTANT: Compile with ONLY optimizer, no loss=:
        vae.compile(optimizer=tf.keras.optimizers.Adam())
    The loss is computed from within train_step / test_step.

    Attributes
    ----------
    encoder : Keras Model — q(z|x), outputs z_mean, z_log_var, z
    decoder : Keras Model — p(x|z), outputs reconstructed image
    beta    : float — KL weight (β-VAE: higher β → more disentangled latent space)

    Loss trackers (for history dict compatible with plot_loss_curves):
        total_loss, recon_loss, kl_loss
    """
    def __init__(self, encoder, decoder, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta    = beta   # β-VAE support

        # Keras metric trackers — updated each step, averaged per epoch
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """Expose trackers to Keras so they are reset each epoch correctly."""
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Custom training step: one forward pass + gradient update.

        The VAE loss (ELBO) is:
            L = recon_loss + β · kl_loss

        recon_loss: binary cross-entropy between original x and reconstructed x̂,
          summed over all pixels and averaged over the batch.
          Using sum over pixels (not mean) provides a balanced scale vs kl_loss.

        kl_loss: KL divergence between learned posterior q(z|x)=N(μ,σ²) and
          the prior p(z)=N(0,1), summed over latent dims and averaged over batch.
          Formula: KL = -½ Σ_j (1 + log(σ_j²) - μ_j² - σ_j²)
          This regularises the latent space to be structured and smooth.

        Why β-VAE?
          β > 1 increases the KL weight, forcing the encoder to use LESS capacity
          in each latent dimension → more sparse, disentangled representations.
          β = 4 is a common choice for learning disentangled factors.
        """
        # FIX 1: Keras always passes data as (x_batch, y_batch) from model.fit().
        # For autoencoders X == y, but the tuple is still created.
        # Passing the whole tuple to self.encoder would cause:
        #   ValueError: Layer "VAE_Encoder" expects 1 input(s), but received 2.
        # Unpack here and use only x (the image tensor).
        if isinstance(data, (tuple, list)):
            x = data[0]   # image tensor — y is identical, not needed
        else:
            x = data      # called directly with a tensor (e.g. in tests)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)   # x, not data
            reconstruction       = self.decoder(z, training=True)

            # ── Reconstruction loss ───────────────────────────────────────────
            # BCE(x, x̂) measures how well each pixel is reconstructed.
            # reduce_sum over (H, W) axes = total bits; reduce_mean over batch.
            # FIX: use x (unpacked image tensor), NOT data (the (x,y) tuple)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2)   # sum per image, not mean, to balance with KL
                )
            )

            # ── KL divergence ─────────────────────────────────────────────────
            # -½ Σ (1 + log_var - μ² - exp(log_var))
            # When posterior = prior (μ=0, σ=1): KL = 0 → no regularisation penalty
            # When posterior collapses (σ→0): KL → ∞ → forces spread-out latent space
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1    # sum over latent dimensions
                )
            )

            # ── Total ELBO loss ───────────────────────────────────────────────
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss":    self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Validation step — same fix as train_step: unpack (x, y) tuple first."""
        # FIX: Keras wraps validation data in (x_batch, y_batch) tuples too.
        # Passing the tuple directly to self.encoder crashes with:
        #   ValueError: Layer "VAE_Encoder" expects 1 input(s), but received 2.
        if isinstance(data, (tuple, list)):
            x = data[0]   # image tensor — only input needed
        else:
            x = data

        z_mean, z_log_var, z = self.encoder(x, training=False)   # FIX: x not data
        reconstruction       = self.decoder(z, training=False)

        # FIX: use x not data in the loss computation
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, reconstruction),
                axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        total_loss = recon_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss":    self.kl_loss_tracker.result(),
        }


def build_and_compile_vae(
    input_shape: tuple = (28, 28, 1),
    latent_dim: int    = 8,
    beta: float        = 1.0,
    learning_rate: float = 1e-3,
    optimizer_name: str  = "Adam",
) -> tuple:
    """
    Convenience function: build encoder + decoder, wrap in VAE, compile, return all.

    IMPORTANT: vae.compile() is called WITHOUT loss= argument.
    The VAE implements its own train/test steps that compute the ELBO.
    Passing loss= to a model with a custom train_step would be ignored (Keras
    doesn't use the compiled loss when train_step is overridden), but can
    cause confusing warnings or errors in some TF versions.

    Returns
    -------
    (vae_model, encoder, decoder)  — vae_model is ready for .fit()
    """
    encoder, decoder  = build_vae(input_shape=input_shape, latent_dim=latent_dim)
    vae               = VAE(encoder, decoder, beta=beta, name="VAE")
    optimizer         = _build_optimizer(optimizer_name, learning_rate)
    # ── Compile WITHOUT loss= ─────────────────────────────────────────────────
    # Loss is handled entirely inside VAE.train_step / VAE.test_step via the
    # ELBO formula.  DO NOT add loss=... here.
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder


# ─────────────────────────────────────────────────────────────────────────────
# 3. DENOISING AUTOENCODER  (MSE, noisy input → clean target)
# ─────────────────────────────────────────────────────────────────────────────

def add_noise(X: np.ndarray, noise_factor: float = 0.3) -> np.ndarray:
    """
    Add Gaussian noise to images and clip to [0, 1].

    Denoising Loss: L = MSE(clean_x, Decoder(Encoder(noisy_x)))
    The model must recover the clean signal from a corrupted version.
    This forces the encoder to learn robust, noise-invariant features.

    Parameters
    ----------
    X            : (N, H, W, C) float32 images in [0, 1]
    noise_factor : standard deviation of Gaussian noise (0.1=mild, 0.5=heavy)

    Returns
    -------
    Noisy images clipped to [0, 1]
    """
    noisy = X + noise_factor * np.random.randn(*X.shape).astype("float32")
    return np.clip(noisy, 0.0, 1.0)


def build_denoising_ae(
    input_shape: tuple = (28, 28, 1),
    latent_dim: int    = 16,
    learning_rate: float = 1e-3,
    optimizer_name: str  = "Adam",
) -> tuple:
    """
    Build a Denoising Autoencoder — same architecture as the standard AE,
    but trained with (noisy_input, clean_target) pairs.

    The denoising objective acts as a regulariser: the model cannot simply
    learn the identity function (it would output noise).  It must extract
    the underlying structure of the digit.

    Training setup (handled in _train_exp5):
        X_train_noisy = add_noise(X_train, noise_factor)
        trainer = Trainer(model, X_train_noisy, X_train_clean)
        # Model predicts clean from noisy — loss = MSE(pred, clean)

    Parameters
    ----------
    input_shape    : (H, W, C) — (28, 28, 1) for MNIST
    latent_dim     : bottleneck dimension
    learning_rate  : optimizer learning rate
    optimizer_name : "Adam" | "SGD" | "RMSprop"

    Returns
    -------
    (autoencoder, encoder, decoder) — same architecture as build_autoencoder()
    """
    # Denoising AE uses the same Conv architecture as the standard AE.
    # The difference is purely in training: noisy input → clean target.
    # build_autoencoder with loss_name="MSE" gives us MSE(recon, clean_target).
    return build_autoencoder(
        input_shape    = input_shape,
        latent_dim     = latent_dim,
        learning_rate  = learning_rate,
        optimizer_name = optimizer_name,
        loss_name      = "MSE",
    )


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    return keras.optimizers.Adam(learning_rate=lr)
