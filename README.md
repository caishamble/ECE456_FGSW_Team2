# Targeted Adversarial Attacks on MNIST Using FGSM

> ECE456 Final Project — White-box targeted evasion attacks on a feed-forward MNIST classifier using the Fast Gradient Sign Method (FGSM).

## Overview

This project investigates the adversarial robustness of a simple Multi-Layer Perceptron (MLP) trained on the MNIST handwritten digit dataset. We implement and evaluate the **Fast Gradient Sign Method (FGSM)** under a white-box, targeted threat model, characterize accuracy degradation across a range of perturbation budgets ε, and compare results against Projected Gradient Descent (PGD).

---

## Key Results

| ε | Accuracy (%) | Accuracy Drop (pp) |
|---|---|---|
| 0.000 | 97.63 | — |
| 0.050 | 61.88 | 35.75 |
| 0.100 | 13.44 | 84.19 |
| 0.150 | 2.60  | 95.03 |
| 0.300 | 0.37  | 97.26 |

- At **ε = 0.05** (imperceptible to humans), accuracy collapses from 97.63% → 61.88%
- At **ε = 0.1**, the model is effectively defeated (~13% accuracy)
- **Digit 9** is the most vulnerable class (99.0% attack success rate)
- **Digit 6** is the most robust class (60.0% attack success rate)
- The model misclassifies adversarial inputs **confidently** — standard confidence-thresholding defenses fail

---

## Model Architecture

```
Input (784)  →  Dense(128, ReLU)  →  Dense(10, Softmax)
```

- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Training Accuracy:** ~98.5%  
- **Validation Accuracy:** ~97.4%  
- **Epochs:** 5  

---

## Attack Method — FGSM

FGSM perturbs an input in the direction that maximally increases the loss:

```
x* = x + ε · sign(∇ₓ L(f(x), y))
```

- **Threat model:** White-box, targeted, ℓ∞-bounded
- **Target:** Cause misclassification into a chosen class y* ≠ y
- **Budget:** Pixels clipped to [0, 1], perturbation bounded by ε

---

## Repository Structure

```
├── train.py                   # Baseline model training
├── test.py                    # Baseline model training result testing
├── FGSM_attack.py             # FGSM attack implementation
├── FGSM_vs_PGD.py             # FGSM and PGD attack compare
├── my_mnist_model.keras/      # Saved model weights
├── results/                   # Output charts and figures
│   ├── chart1_accuracy_vs_epsilon.png
│   ├── chart2_per_digit_vulnerability.png
│   ├── chart3_confidence_distribution.png
│   └── chart4_fgsm_vs_pgd.png
└── report/                    # LaTeX source for the final report
```

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Train the Baseline Model

```bash
python train.py
```

### Test the Baseline Model

```bash
python test.py
```

### Run FGSM Attack

```bash
python FGSM_attack.py
```

### Compare FGSM vs. PGD

```bash
python FGSM_vs_PGD.py
```

---

## FGSM vs. PGD Comparison

| ε | FGSM Accuracy (%) | PGD Accuracy (%) |
|---|---|---|
| 0.05 | 56.2 | 47.8 |
| 0.10 | 10.2 | 6.7  |

PGD is the stronger attack at small, imperceptible ε values, confirming FGSM underestimates the model's true vulnerability (Madry et al., 2018).

---

## References

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018
- Carlini & Wagner, *Evaluating the Robustness of Neural Networks*, IEEE S&P 2017
- Kurakin et al., *Adversarial Examples in the Physical World*, ICLR Workshop 2017
- LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998

---

## Team

| Name | Contribution |
|---|---|
| Xiangbo Cai | Model Training, Conceptualization |
| Daniel Liu | FGSW Model attacking, Writing|
| Everett Edwards | Model Training, Conceptualization|
| Sardar Rafayet Bin Murtaza | FGSW Model attacking, Writing|
| Hussain Alabbad | FGSW Model attacking, Writing|
| Matthew Atkinson | Writing, Literature review|

---

*ECE456 — Machine Learning Security*
