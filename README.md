# Neural networks — from one to two layers

Simple neural networks implemented **from scratch in Python/NumPy**, together with an
extensive presentation that explains both the theory and how the code works — from
logistic regression (single-layer network) to error backpropagation (two-layer network,
the XOR problem).

## Repository contents

| File | Description |
| --- | --- |
| `neural_networks_presentation_PL_EN.html` | Full **bilingual** presentation (PL/EN, with a language switch): theory → picture → code reference (neuron, sigmoid, delta rule, backpropagation, full training steps worked out numerically). |
| `single_layer_network_in_python.ipynb` | Notebook (Google Colab) — **single-layer** network (classification, delta rule). |
| `two_layer_network_in_python.ipynb` | Notebook (Google Colab) — **two-layer** network (hidden layer, backpropagation, XOR). |

## Open in Google Colab

- Single-layer network: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finchWro/NeuralNetwork/blob/main/single_layer_network_in_python.ipynb)
- Two-layer network: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finchWro/NeuralNetwork/blob/main/two_layer_network_in_python.ipynb)

## Presentation

Open `neural_networks_presentation_PL_EN.html` in your browser. The presentation is
**bilingual** — a `PL / EN` switch is in the top-right corner. It covers, among other things:

- **Part 1 — single-layer network:** the biological neuron as inspiration, the transition
  from linear regression to the sigmoid, architecture and weight initialization, the
  forward pass, error and cost (MSE), the delta rule, and one full training step worked
  out numerically.
- **Part 2 — two-layer network:** why a hidden layer is needed (XOR), the role of the
  bias, the forward pass through two layers, the idea and full formulas of backpropagation,
  and one complete training step worked out numerically.

## Requirements

- Python 3 and `numpy` (the core of the implementation).
- `matplotlib` — plots (e.g. the MSE curve).

```bash
pip install numpy matplotlib
```

## Author

Błażej Zięba — 15.06.2026
