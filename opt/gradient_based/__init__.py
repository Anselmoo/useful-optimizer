"""Gradient-based optimization algorithms.

This module contains optimizers that use gradient information to find optimal solutions.
Includes: AdaDelta, AdaGrad, AdaMax, AdamW, Adam, AMSGrad, NAdam, Nesterov, RMSprop, SGD.
"""

from __future__ import annotations

from opt.gradient_based.adadelta import AdaDelta
from opt.gradient_based.adagrad import ADAGrad
from opt.gradient_based.adamax import AdaMax
from opt.gradient_based.adamw import AdamW
from opt.gradient_based.adaptive_moment_estimation import ADAMOptimization
from opt.gradient_based.amsgrad import AMSGrad
from opt.gradient_based.nadam import Nadam
from opt.gradient_based.nesterov_accelerated_gradient import NesterovAcceleratedGradient
from opt.gradient_based.rmsprop import RMSprop
from opt.gradient_based.sgd_momentum import SGDMomentum
from opt.gradient_based.stochastic_gradient_descent import SGD


__all__: list[str] = [
    "SGD",
    "ADAGrad",
    "ADAMOptimization",
    "AMSGrad",
    "AdaDelta",
    "AdaMax",
    "AdamW",
    "Nadam",
    "NesterovAcceleratedGradient",
    "RMSprop",
    "SGDMomentum",
]
