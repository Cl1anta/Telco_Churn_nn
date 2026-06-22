# Telco Customer Churn Prediction — From-Scratch Neural Network

This is a project built as part of the **Neural Networks track of Summer of Science (SoS) 2025–26**, mentored by Aryan B Suryawanshi. The goal is to deeply understand how neural networks learn, not just to build a classifier that works, but to build one from scratch, understand every line of the math behind it, and compare it honestly against a strong classical ML baseline.

---------------------------------------------------

## What This Project Is About

Customer churn — when a subscriber leaves for a competitor — is one of the most costly problems in the telecom industry. Retaining an existing customer is far cheaper than acquiring a new one, which makes early, accurate churn identification genuinely valuable.

This project uses the **IBM Telco Customer Churn dataset** (7,043 customers, 26.5% churn rate) and approaches the problem in two stages:

**Stage 1 — Baseline (XGBoost):** A production-style gradient-boosted tree pipeline with tuned hyperparameters and class-weighting to handle the label imbalance. This sets a strong, realistic benchmark.

Reference for model: [XGBoost model](https://github.com/koladefaj/Telco-churn-prediction/blob/main/README.md?plain=1)

**Stage 2 — From-Scratch Neural Network (NumPy only):** A feedforward neural network implemented entirely in NumPy — no PyTorch, no TensorFlow, no autograd. Forward propagation, backpropagation, He initialisation, a numerically stable class-weighted BCE loss, and four gradient-descent optimizers all derived and coded by hand. The NN is then benchmarked against the XGBoost baseline.

The central research question is **"how do different gradient-descent optimizers compare on the same problem, and why?"**

---

## Project Structure

```
Telco-churn-prediction/
│
├── Telco-Churn.csv      # IBM Telco Customer Churn dataset          
├── Telco-nn.py          #the neural network model that includes all the EDA
├── Project report       #details of the entire project building process and outputs.
├── telco_optimisation_comparison        #comares optimisation of GD methods
└── README.md
```

---

## The Neural Network — What's Built From Scratch

Everything in `telco_nn.py` is implemented manually in NumPy. No autodiff, no `.backward()`.

| Component | Implementation |
|---|---|
| Architecture | Input → 16 → 8 → 1 (fully connected) |
| Activations | ReLU (hidden), Sigmoid (output) |
| Weight init | He initialisation (`√(2 / fan_in)`) |
| Loss | Numerically stable, class-weighted BCE (log-sum-exp form) |
| Class imbalance | `pos_weight ≈ 2.77` (non-churn/churn ratio in training set) |
| Backprop | Manually derived gradients, layer by layer |
| Gradient safety | Clipping to `[-2, 2]`, NaN zeroing |

### Optimizers Compared

Four optimizers are trained on identical data splits and architecture, the only variable being the update rule:

| Optimizer | Batch Size | Learning Rate |
|---|---|---|
| Batch Gradient Descent | Full dataset (5,088) | 0.01 |
| Mini-Batch GD | 64 | 0.005 |
| SGD + Momentum (β=0.9) | 64 | 0.005 |
| Adam (β₁=0.9, β₂=0.999) | 64 | 0.0005 |

---

## Key Results

| Model / Optimizer | Test ROC-AUC |
|---|---|
| XGBoost Baseline (tuned) | **0.8468** |
| Neural Network — Adam | 0.8431 |
| Neural Network — SGD + Momentum | 0.8410 |
| Neural Network — Mini-Batch GD | 0.8408 |
| Neural Network — Batch GD | 0.7548 *(not converged at 80 epochs)* |

The best from-scratch network (Adam) comes within **0.4 AUC points** of the tuned XGBoost ensemble and matches it exactly on F1-score (0.64) at each model's optimal threshold — despite using only 625 trainable parameters and no regularisation.

Batch GD's poor showing isn't a bug — it's a deliberate illustration of **why update frequency matters**: with a full-batch size, it makes only 80 parameter updates across 80 epochs versus ~6,400 for the mini-batch methods.

---

## EDA Highlights

A few patterns that stood out in the data and directly shape the modelling approach:

- **Contract type** is the strongest single predictor: month-to-month customers churn at 42.7% vs. 2.8% for two-year contract holders.
- **Tenure** follows a steep gradient: first-year customers churn at 48.3%, dropping to 11.1% for customers past four years.
- **Fiber optic** internet and **electronic check** payment both show unusually high churn (~42–45%), likely proxies for a specific high-risk customer segment.
- The 26.5% churn rate is a meaningful imbalance — naive accuracy would be misleading, which is why **ROC-AUC** is used as the primary metric throughout.

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost joblib
```

### Dataset

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it at `data/Telco-Churn.csv`.

### Run the XGBoost baseline

```bash
python src/train_model.py
```

### Run the neural network optimizer comparison

```bash
python src/telco_nn.py
```

This trains all four optimizers and saves the convergence plot to `outputs/telco_optimization_comparison.png`.

---



Main references: [IIT Madras Deep Learning lectures](https://youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM), [Andrew Ng's Deep Learning Specialization](https://youtu.be/CS4cs9xVecg), [Andrej Karpathy's micrograd series](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), and *Deep Learning* by Goodfellow, Bengio & Courville.

---

## Acknowledgements

Thanks to Aryan B Suryawanshi for the weekly guidance and resource curation throughout the track.
