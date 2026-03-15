from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .activations import activation_functions


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class ParameterSpec:
    name: str
    shape: Tuple[int, ...]
    size: int


class DensityMatrixAnsatz:
    parameter_dtype = np.float64

    def log_psi(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> complex:
        raise NotImplementedError

    def log_psi_batch(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> np.ndarray:
        return np.asarray([self.log_psi(r, c) for r, c in zip(sigma_row, sigma_col)], dtype=np.complex128)

    def derivatives(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters_vector(self) -> np.ndarray:
        chunks = []
        for spec in self.parameter_specs():
            value = getattr(self, spec.name)
            chunks.append(np.asarray(value, dtype=self.parameter_dtype).reshape(-1))
        return np.concatenate(chunks)

    def set_parameters_vector(self, flat: np.ndarray) -> None:
        flat = np.asarray(flat, dtype=self.parameter_dtype)
        start = 0
        for spec in self.parameter_specs():
            stop = start + spec.size
            value = flat[start:stop].reshape(spec.shape)
            setattr(self, spec.name, value.copy())
            start = stop

    def apply_update(self, delta: np.ndarray, learning_rate: float) -> None:
        params = self.parameters_vector()
        params = params - learning_rate * np.asarray(delta, dtype=self.parameter_dtype)
        self.set_parameters_vector(params)

    def parameter_specs(self) -> Iterable[ParameterSpec]:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return sum(spec.size for spec in self.parameter_specs())


class RBMSplit(DensityMatrixAnsatz):
    parameter_dtype = np.complex128

    def __init__(self, n_visible: int, alpha: float, seed: Optional[int] = None) -> None:
        n_hidden = int(round(alpha * n_visible))
        rng = _rng(seed)
        self.ar = 0.01 * (rng.standard_normal(n_visible) + 1j * rng.standard_normal(n_visible))
        self.ac = 0.01 * (rng.standard_normal(n_visible) + 1j * rng.standard_normal(n_visible))
        self.b = 0.05 * (rng.standard_normal(n_hidden) + 1j * rng.standard_normal(n_hidden))
        self.Wr = 0.01 * (rng.standard_normal((n_hidden, n_visible)) + 1j * rng.standard_normal((n_hidden, n_visible)))
        self.Wc = 0.01 * (rng.standard_normal((n_hidden, n_visible)) + 1j * rng.standard_normal((n_hidden, n_visible)))

    def parameter_specs(self) -> Iterable[ParameterSpec]:
        for name in ("ar", "ac", "b", "Wr", "Wc"):
            value = getattr(self, name)
            yield ParameterSpec(name=name, shape=value.shape, size=value.size)

    def log_psi(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> complex:
        theta = self.b + self.Wr @ sigma_row + self.Wc @ sigma_col
        return np.dot(self.ar, sigma_row) + np.dot(self.ac, sigma_col) + np.sum(np.log(2.0 * np.cosh(theta)))

    def derivatives(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> np.ndarray:
        theta = self.b + self.Wr @ sigma_row + self.Wc @ sigma_col
        tanh_theta = np.tanh(theta)
        return np.concatenate(
            [
                sigma_row.astype(np.complex128),
                sigma_col.astype(np.complex128),
                tanh_theta,
                np.outer(tanh_theta, sigma_row).reshape(-1),
                np.outer(tanh_theta, sigma_col).reshape(-1),
            ]
        )


class NeuralDensityMatrix(DensityMatrixAnsatz):
    parameter_dtype = np.float64

    def __init__(
        self,
        n_visible: int,
        alpha_hidden: float,
        alpha_ancilla: float,
        activation: str = "softplus",
        seed: Optional[int] = None,
    ) -> None:
        if activation not in activation_functions:
            raise ValueError("Unsupported activation: %s" % activation)
        self.activation_name = activation
        self.f, self.f_prime = activation_functions[activation]
        n_hidden = int(round(alpha_hidden * n_visible))
        n_ancilla = int(round(alpha_ancilla * n_visible))
        rng = _rng(seed)
        self.b_p = 0.005 * rng.standard_normal(n_visible)
        self.h_p = 0.005 * rng.standard_normal(n_hidden)
        self.W_p = 0.01 * rng.standard_normal((n_hidden, n_visible))
        self.U_p = 0.01 * rng.standard_normal((n_ancilla, n_visible))
        self.b_m = 0.005 * rng.standard_normal(n_visible)
        self.h_m = 0.005 * rng.standard_normal(n_hidden)
        self.d_p = 0.005 * rng.standard_normal(n_ancilla)
        self.W_m = 0.01 * rng.standard_normal((n_hidden, n_visible))
        self.U_m = 0.01 * rng.standard_normal((n_ancilla, n_visible))

    def parameter_specs(self) -> Iterable[ParameterSpec]:
        for name in ("b_p", "h_p", "W_p", "U_p", "b_m", "h_m", "d_p", "W_m", "U_m"):
            value = getattr(self, name)
            yield ParameterSpec(name=name, shape=value.shape, size=value.size)

    def log_psi(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> complex:
        theta_p_row = self.h_p + self.W_p @ sigma_row
        theta_m_row = self.h_m + self.W_m @ sigma_row
        theta_p_col = self.h_p + self.W_p @ sigma_col
        theta_m_col = self.h_m + self.W_m @ sigma_col
        sigma_plus = sigma_row + sigma_col
        sigma_minus = sigma_row - sigma_col
        ancilla_theta = 0.5 * (self.U_p @ sigma_plus) + 0.5j * (self.U_m @ sigma_minus) + self.d_p

        phi_p = 0.5 * (
            np.sum(self.f(theta_p_row))
            + np.sum(self.f(theta_p_col))
            + np.dot(self.b_p, sigma_plus)
        )
        phi_m = 0.5j * (
            np.sum(self.f(theta_m_row))
            - np.sum(self.f(theta_m_col))
            + np.dot(self.b_m, sigma_minus)
        )
        phi_a = np.sum(self.f(ancilla_theta))
        return complex(phi_p + phi_m + phi_a)

    def derivatives(self, sigma_row: np.ndarray, sigma_col: np.ndarray) -> np.ndarray:
        theta_p_row = self.h_p + self.W_p @ sigma_row
        theta_m_row = self.h_m + self.W_m @ sigma_row
        theta_p_col = self.h_p + self.W_p @ sigma_col
        theta_m_col = self.h_m + self.W_m @ sigma_col
        sigma_plus = sigma_row + sigma_col
        sigma_minus = sigma_row - sigma_col
        ancilla_theta = 0.5 * (self.U_p @ sigma_plus) + 0.5j * (self.U_m @ sigma_minus) + self.d_p

        f_p_row_prime = self.f_prime(theta_p_row)
        f_m_row_prime = self.f_prime(theta_m_row)
        f_p_col_prime = self.f_prime(theta_p_col)
        f_m_col_prime = self.f_prime(theta_m_col)
        f_a_prime = self.f_prime(ancilla_theta)

        parts = [
            0.5 * sigma_plus,
            0.5 * (f_p_row_prime + f_p_col_prime),
            0.5 * (np.outer(f_p_row_prime, sigma_row) + np.outer(f_p_col_prime, sigma_col)).reshape(-1),
            0.5 * np.outer(f_a_prime, sigma_plus).reshape(-1),
            0.5j * sigma_minus,
            0.5j * (f_m_row_prime - f_m_col_prime),
            f_a_prime,
            0.5j * (np.outer(f_m_row_prime, sigma_row) - np.outer(f_m_col_prime, sigma_col)).reshape(-1),
            0.5j * np.outer(f_a_prime, sigma_minus).reshape(-1),
        ]
        return np.concatenate([np.asarray(part, dtype=np.complex128).reshape(-1) for part in parts])
