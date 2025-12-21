---
title: Gradient-Based Optimizers
description: Algorithms using derivative information for optimization
tags:
  - gradient-based
  - deep-learning
---

# Gradient-Based Optimizers

Gradient-based optimizers use derivative information to guide the search toward optimal solutions. These are the workhorses of modern machine learning and deep learning.

---

## Overview

These algorithms follow the gradient (direction of steepest descent) to find minima:

\[
\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)
\]

Key concepts:

- **Learning rate** (\(\alpha\)) - Step size in the descent direction
- **Momentum** - Accumulating past gradients for smoother convergence
- **Adaptive learning** - Per-parameter learning rate adjustment

---

## Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **SGD** | Basic stochastic gradient descent | Simple problems |
| **SGD Momentum** | SGD with momentum term | Faster convergence |
| **Nesterov** | Lookahead momentum | Improved momentum |
| **Adagrad** | Per-parameter learning rates | Sparse features |
| **Adadelta** | No learning rate needed | Avoiding lr tuning |
| **RMSprop** | Moving average of gradients | Non-stationary |
| **Adam** | Adaptive moments | General purpose |
| **AdamW** | Adam with weight decay | Regularization |
| **AdaMax** | Adam with infinity norm | Large gradients |
| **Nadam** | Nesterov + Adam | Fast convergence |
| **AMSGrad** | Non-decreasing moments | Convergence guarantee |

---

## Usage Pattern

```python
from opt.gradient_based import AdamW
from opt.benchmark.functions import rosenbrock

optimizer = AdamW(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    learning_rate=0.01,
    max_iter=1000,
)

best_solution, best_fitness = optimizer.search()
```

---

## Learning Rate Guidelines

| Algorithm | Typical Learning Rate |
|-----------|----------------------|
| SGD | 0.01 - 0.1 |
| SGD Momentum | 0.001 - 0.01 |
| Adam/AdamW | 0.0001 - 0.001 |
| RMSprop | 0.001 - 0.01 |

---

## When to Use

!!! success "Good For"
    - Smooth, differentiable functions
    - Large-scale optimization
    - Deep learning / neural networks
    - Fast convergence needed

!!! warning "Limitations"
    - Require differentiable objectives
    - May converge to local minima
    - Sensitive to learning rate
    - May struggle with multimodal landscapes
