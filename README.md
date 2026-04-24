# 🧪 Interactive Loss Function Virtual Lab

> A university-level deep learning teaching tool built with Streamlit, TensorFlow, and Plotly.

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Navigate into the app directory
cd virtual_lab

# 3. Launch the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
virtual_lab/
├── app.py                   ← Main Streamlit entry point
├── requirements.txt
├── config.py                ← Constants, color palettes, loss metadata, LaTeX formulas
├── datasets/
│   ├── mnist_loader.py      ← MNIST via keras.datasets
│   ├── binary_loader.py     ← Breast Cancer (sklearn)
│   ├── regression_loader.py ← California Housing + outlier injection
│   └── fraud_loader.py      ← Synthetic imbalanced fraud dataset
├── models/
│   ├── mlp.py               ← Multi-Layer Perceptron (configurable)
│   ├── cnn.py               ← Small CNN for MNIST
│   └── autoencoder.py       ← Conv Autoencoder with 2D bottleneck
├── losses/
│   ├── focal_loss.py        ← Custom Keras Focal Loss implementation
│   └── loss_registry.py     ← Friendly-name → Keras loss mapping
├── training/
│   └── trainer.py           ← Keras fit() + Streamlit callbacks + gradient recorder
├── plots/
│   ├── loss_curves.py       ← Animated training/validation curves
│   ├── confusion_matrix.py  ← Interactive confusion matrix heatmap
│   ├── prediction_viz.py    ← Image grid, scatter, probability bars
│   ├── roc_curve.py         ← ROC + Precision-Recall curves
│   ├── loss_landscape.py    ← 3D loss surface + optimizer path
│   └── gradient_viz.py      ← Gradient magnitude per layer
└── utils/
    ├── metrics.py           ← Unified metrics computation
    └── explainer.py         ← Auto-generated educational content
```

---

## 🔬 Experiments

| # | Name | Dataset | Loss Functions | Key Insight |
|---|------|---------|----------------|-------------|
| 1 | Multi-class Classification | MNIST | Cross Entropy vs MSE | Why CE crushes MSE for classification |
| 2 | Binary Classification | Breast Cancer (sklearn) | Binary CE | Threshold slider → live confusion matrix |
| 3 | Regression + Outlier Robustness | California Housing | MSE vs Huber | How MSE breaks under outliers |
| 4 | Imbalanced Classification | Synthetic Fraud | BCE vs Focal Loss | Accuracy/F1 paradox, minority class rescue |
| 5 | Autoencoder Reconstruction | MNIST | MSE (pixel-wise) | Latent space 2D scatter, reconstruction quality |

---

## 🖥️ UI Tabs

| Tab | Contents |
|-----|----------|
| 📊 Training Dashboard | Loss formula, animated training curves, final metrics |
| 🔍 Predictions | MNIST grid, prob bars, regression scatter, AE reconstructions |
| 📉 Loss Analysis | Confusion matrix, ROC, PR curve, gradient magnitude chart |
| 🌋 Loss Landscape | Interactive 3D loss surface with optimizer trajectory |
| ⚖️ Compare Mode | Side-by-side losses with winner badge |
| 💡 Explainer | Educational content: formula, usage guide, viva tips |

---

## 🎛️ Sidebar Controls

- **Experiment** selector (1–5)
- **Model** type (MLP / CNN / Dense NN / Autoencoder — context-aware)
- **Loss function** selector (context-aware per experiment)
- **Comparison Mode** toggle (train 2 losses simultaneously)
- **Optimizer**: Adam / SGD / RMSprop
- **Epochs**: 1–50
- **Batch size**: 32 / 64 / 128 / 256
- **Learning rate**: 1e-4 → 1e-1 (log scale)
- **Dropout**: 0.0 – 0.5
- **Experiment-specific**: outlier sliders (Exp 3), fraud ratio (Exp 4), focal α/γ (Exp 4)
- **Threshold slider** (Exp 2): updates metrics live without re-training

---

## 📚 Advanced Features

1. **Outlier Injection** (Exp 3): Slider injects synthetic outliers into training labels
2. **Imbalance Ratio** (Exp 4): Slider controls minority class percentage
3. **Focal Loss Controls** (Exp 4): α and γ sliders with real-time explanation
4. **Loss Landscape** (Tab 4): 30×30 grid computed by perturbing two weight directions
5. **Gradient Tracker**: Records ‖∇‖ per layer per epoch using a custom Keras callback
6. **Dataset Preview**: First 9 images, class distributions, scatter plots

---

## 🔧 System Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for CNN training)
- No GPU required — all experiments run on CPU
- No external API keys needed

---

## 📖 Educational Notes

This lab is designed as a hands-on supplement for university deep learning courses.
Each module contains inline teaching comments explaining:

- **Why** each architectural decision was made
- **What** the mathematical implications are
- **When** each loss function applies (and when it fails)

Students can modify `config.py` to add new loss functions to the registry
and `datasets/` to plug in their own datasets without touching `app.py`.

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `streamlit run app.py` from inside `virtual_lab/` |
| Slow CNN training | Reduce epochs or use MLP model type |
| Memory error | Reduce batch size or use smaller subset in `mnist_loader.py` |
| TF GPU error | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |

---

*Built for deep learning education. Inspect the source code — it was written to be read.*
