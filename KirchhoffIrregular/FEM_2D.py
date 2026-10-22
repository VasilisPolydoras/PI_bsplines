"""Morley FEM reference for a Kirchhoff plate with a free circular hole."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from skfem import Basis, BilinearForm, LinearForm, MeshTri, asm, condense, solve
    from skfem.element import ElementTriMorley
    from skfem.helpers import dd, ddot, trace
except ImportError as error:
    raise ImportError(
        "FEM_2D.py requires scikit-fem. Install it with "
        "'python -m pip install scikit-fem'."
    ) from error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KirchhoffIrregular.Bspline_2D import (
    FLEXURAL_RIGIDITY,
    HOLE_CENTER,
    HOLE_RADIUS,
    LOAD_AMPLITUDE,
    PLATE_LENGTH,
    POISSON_RATIO,
)


DEFAULT_CELLS_PER_DIRECTION = 100
RESULT_PATH = (
    Path(__file__).resolve().parent / "results" / "fem_reference.npz"
)


def create_mesh(cells_per_direction: int) -> MeshTri:
    """Create a body-fitted triangular mesh between circle and square."""
    if cells_per_direction < 10:
        raise ValueError("cells_per_direction must be at least 10")
    n_theta = max(40, 4 * cells_per_direction)
    n_theta += (-n_theta) % 8  # include all four square corners
    n_radial = max(5, cells_per_direction // 2)
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    cosine = np.cos(theta)
    sine = np.sin(theta)
    half_width = PLATE_LENGTH / 2.0
    outer_radius = half_width / np.maximum(np.abs(cosine), np.abs(sine))

    points = []
    for radial_index in range(n_radial + 1):
        fraction = radial_index / n_radial
        radius = HOLE_RADIUS + fraction * (outer_radius - HOLE_RADIUS)
        points.append(
            np.column_stack(
                (
                    HOLE_CENTER[0] + radius * cosine,
                    HOLE_CENTER[1] + radius * sine,
                )
            )
        )
    points = np.concatenate(points, axis=0)

    triangles = []
    for radial_index in range(n_radial):
        inner_offset = radial_index * n_theta
        outer_offset = (radial_index + 1) * n_theta
        for angular_index in range(n_theta):
            next_index = (angular_index + 1) % n_theta
            inner_left = inner_offset + angular_index
            inner_right = inner_offset + next_index
            outer_left = outer_offset + angular_index
            outer_right = outer_offset + next_index
            triangles.append((inner_left, outer_left, outer_right))
            triangles.append((inner_left, outer_right, inner_right))
    return MeshTri(points.T, np.asarray(triangles, dtype=np.int32).T)


@BilinearForm
def plate_bilinear_form(u, v, _):
    return FLEXURAL_RIGIDITY * (
        (1.0 - POISSON_RATIO) * ddot(dd(u), dd(v))
        + POISSON_RATIO * trace(dd(u)) * trace(dd(v))
    )


@LinearForm
def transverse_load(v, w):
    load = (
        LOAD_AMPLITUDE
        * np.sin(np.pi * w.x[0] / PLATE_LENGTH)
        * np.sin(np.pi * w.x[1] / PLATE_LENGTH)
    )
    return load * v


def solve_plate(cells_per_direction: int):
    mesh = create_mesh(cells_per_direction)
    basis = Basis(mesh, ElementTriMorley())
    stiffness = asm(plate_bilinear_form, basis)
    force = asm(transverse_load, basis)

    # Constrain w only on the four simply supported outer edges.  The outer
    # M_nn=0 condition and both free-hole conditions are natural conditions.
    outer_facets = mesh.facets_satisfying(
        lambda x: (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], PLATE_LENGTH)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], PLATE_LENGTH)
        )
    )
    boundary_dofs = basis.get_dofs(outer_facets).nodal["u"]
    solution = solve(*condense(stiffness, force, D=boundary_dofs))
    nodal_deflection = solution[basis.nodal_dofs[0]]
    return mesh, nodal_deflection


def save_reference(
    mesh: MeshTri,
    nodal_deflection: np.ndarray,
    cells_per_direction: int,
    path: Path = RESULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=mesh.p.T,
        triangles=mesh.t.T,
        deflection=nodal_deflection,
        cells_per_direction=np.asarray(cells_per_direction),
        plate_length=np.asarray(PLATE_LENGTH),
        hole_center=np.asarray(HOLE_CENTER),
        hole_radius=np.asarray(HOLE_RADIUS),
        load_amplitude=np.asarray(LOAD_AMPLITUDE),
        flexural_rigidity=np.asarray(FLEXURAL_RIGIDITY),
        poisson_ratio=np.asarray(POISSON_RATIO),
        hole_boundary=np.asarray("free"),
    )
    print(f"Saved FEM reference to: {path}")


def plot_reference(mesh: MeshTri, nodal_deflection: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    contour = axis.tricontourf(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        nodal_deflection,
        levels=50,
        cmap="viridis",
    )
    figure.colorbar(contour, ax=axis, label="FEM deflection w")
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Morley FEM reference")
    plt.show()


def main(cells_per_direction: int, show_plot: bool) -> None:
    print(
        f"Solving with mesh resolution parameter {cells_per_direction}..."
    )
    mesh, nodal_deflection = solve_plate(cells_per_direction)
    save_reference(mesh, nodal_deflection, cells_per_direction)
    print(f"Mesh vertices: {mesh.nvertices}")
    print(f"Mesh triangles: {mesh.nelements}")
    print(f"Maximum deflection: {nodal_deflection.max():.6e}")
    print(f"Minimum deflection: {nodal_deflection.min():.6e}")
    if show_plot:
        plot_reference(mesh, nodal_deflection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Morley FEM reference for the irregular plate"
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=DEFAULT_CELLS_PER_DIRECTION,
        help="number of background cells in each coordinate direction",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="save the reference without opening a figure",
    )
    arguments = parser.parse_args()
    main(arguments.cells, not arguments.no_plot)
