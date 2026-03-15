from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class StochasticReconfiguration:
    diagonal_shift: float = 1e-3

    def solve(self, log_derivatives: np.ndarray, local_values: np.ndarray, real_parameters: bool) -> np.ndarray:
        centered_o = log_derivatives - np.mean(log_derivatives, axis=0, keepdims=True)
        centered_l = local_values - np.mean(local_values)
        s_matrix = centered_o.conj().T @ centered_o / len(log_derivatives)
        force = centered_o.conj().T @ centered_l / len(log_derivatives)
        if real_parameters:
            s_matrix = np.real(s_matrix)
            force = np.real(force)
        s_matrix = s_matrix + self.diagonal_shift * np.eye(s_matrix.shape[0], dtype=s_matrix.dtype)
        return linalg.solve(s_matrix, force, assume_a="her")
