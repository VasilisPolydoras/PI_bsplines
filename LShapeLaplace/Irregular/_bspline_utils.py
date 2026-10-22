"""Geometry-independent utilities for tensor-product B-splines."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the random-number generators used by the solver."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    """Resolve the configured CPU/GPU name to a PyTorch device."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'cpu' or 'gpu'")


def basis_matrix(
    values: torch.Tensor, knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Evaluate every univariate B-spline basis function at ``values``."""
    values = values.reshape(-1)
    basis = (
        (values[:, None] >= knots[:-1])
        & (values[:, None] < knots[1:])
    ).to(values.dtype)
    at_right_end = values == knots[-1]
    if at_right_end.any():
        basis[at_right_end] = 0.0
        basis[at_right_end, -1] = 1.0
    for current_degree in range(1, degree + 1):
        column_count = knots.numel() - current_degree - 1
        updated = torch.zeros(
            values.numel(),
            column_count,
            device=values.device,
            dtype=values.dtype,
        )
        for index in range(column_count):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = (
                knots[index + current_degree + 1] - knots[index + 1]
            )
            if left_denominator > 0:
                updated[:, index] += (
                    (values - knots[index]) / left_denominator
                ) * basis[:, index]
            if right_denominator > 0:
                updated[:, index] += (
                    (knots[index + current_degree + 1] - values)
                    / right_denominator
                ) * basis[:, index + 1]
        basis = updated
    if at_right_end.any():
        basis[at_right_end] = 0.0
        basis[at_right_end, -1] = 1.0
    return basis


def basis_derivative_matrix(
    values: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    derivative_order: int,
) -> torch.Tensor:
    """Evaluate one derivative order of every univariate basis function."""
    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")
    n_basis = knots.numel() - degree - 1
    if derivative_order == 0:
        return basis_matrix(values, knots, degree)
    if derivative_order > degree:
        return torch.zeros(
            values.numel(),
            n_basis,
            device=values.device,
            dtype=values.dtype,
        )
    values = values.reshape(-1)
    at_right_end = values == knots[-1]
    if at_right_end.any():
        values = values.clone()
        values[at_right_end] = torch.nextafter(knots[-1], knots[0])
    lower = basis_derivative_matrix(
        values, knots, degree - 1, derivative_order - 1
    )
    result = torch.zeros(
        values.numel(), n_basis, device=values.device, dtype=values.dtype
    )
    for index in range(n_basis):
        left_denominator = knots[index + degree] - knots[index]
        right_denominator = knots[index + degree + 1] - knots[index + 1]
        if left_denominator > 0:
            result[:, index] += degree / left_denominator * lower[:, index]
        if right_denominator > 0:
            result[:, index] -= degree / right_denominator * lower[:, index + 1]
    return result


def greville_abscissae(knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Return one Greville abscissa for every basis function."""
    n_basis = knots.numel() - degree - 1
    return torch.stack(
        [
            knots[index + 1 : index + degree + 1].mean()
            for index in range(n_basis)
        ]
    )


def interpolate_boundary_trace(
    knots: torch.Tensor,
    degree: int,
    physical_points: torch.Tensor,
    solution_function,
) -> torch.Tensor:
    """Interpolate a prescribed trace in a univariate spline space."""
    greville = greville_abscissae(knots, degree)
    interpolation_matrix = basis_matrix(greville, knots, degree)
    return torch.linalg.solve(
        interpolation_matrix, solution_function(physical_points)
    )
