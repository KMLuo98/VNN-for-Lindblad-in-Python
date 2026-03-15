from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(z))


def softplus_prime(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logcosh(z: np.ndarray) -> np.ndarray:
    return np.log(np.cosh(z))


def logcosh_prime(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


activation_functions: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "softplus": (softplus, softplus_prime),
    "logcosh": (logcosh, logcosh_prime),
}
