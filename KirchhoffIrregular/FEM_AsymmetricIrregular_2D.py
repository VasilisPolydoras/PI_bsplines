"""Morley FEM reference for the asymmetric irregular Kirchhoff plate."""

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
        "FEM_AsymmetricIrregular_2D.py requires scikit-fem. Install it with "
        "'python -m pip install scikit-fem'."
    ) from error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KirchhoffIrregular.Bspline_AsymmetricIrregular_2D import (
    FLEXURAL_RIGIDITY,
    HOLE_CENTER,
    HOLE_RADIUS,
    LOAD_AMPLITUDE,
    OUTER_CENTER,
    OUTER_COS3_AMPLITUDE,
    OUTER_MEAN_RADIUS,
    OUTER_SIN2_AMPLITUDE,
    PLATE_LENGTH,
    POISSON_RATIO,
)


DEFAULT_CELLS_PER_DIRECTION = 100
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "fem_asymmetric_irregular_reference.npz"
)


def outer_radius_numpy(theta: np.ndarray) -> np.ndarray:
    """Evaluate the same irregular polar boundary used by the B-spline file."""
    return OUTER_MEAN_RADIUS * (
        1.0
        + OUTER_COS3_AMPLITUDE * np.cos(3.0 * theta)
        + OUTER_SIN2_AMPLITUDE * np.sin(2.0 * theta)
    )


def inside_outer_numpy(points: np.ndarray) -> np.ndarray:
    """Return whether points of shape (..., 2) lie inside the exterior."""
    dx = points[..., 0] - OUTER_CENTER[0]
    dy = points[..., 1] - OUTER_CENTER[1]
    theta = np.arctan2(dy, dx)
    radius = np.sqrt(dx**2 + dy**2)
    return radius < outer_radius_numpy(theta)


def distance_to_outer_boundary(ray_angles: np.ndarray) -> np.ndarray:
    """Intersect rays from the hole center with the irregular exterior.

    The geometry is star-shaped with respect to the selected hole center.
    Vectorized bisection gives a boundary point for every angular ray.
    """
    directions = np.column_stack((np.cos(ray_angles), np.sin(ray_angles)))
    center = np.asarray(HOLE_CENTER)
    lower = np.full(ray_angles.shape, HOLE_RADIUS)
    upper = np.full(ray_angles.shape, 2.0 * PLATE_LENGTH)
    if np.any(inside_outer_numpy(center + upper[:, None] * directions)):
        raise ValueError("Ray-search upper bound must lie outside the domain")

    for _ in range(70):
        middle = 0.5 * (lower + upper)
        middle_points = center + middle[:, None] * directions
        is_inside = inside_outer_numpy(middle_points)
        lower = np.where(is_inside, middle, lower)
        upper = np.where(is_inside, upper, middle)
    return 0.5 * (lower + upper)


def create_mesh(cells_per_direction: int) -> MeshTri:
    """Create a body-fitted ring mesh between the hole and outer curve."""
    if cells_per_direction < 10:
        raise ValueError("cells_per_direction must be at least 10")
    n_theta = max(48, 4 * cells_per_direction)
    n_theta += (-n_theta) % 8
    n_radial = max(6, cells_per_direction // 2)
    ray_angles = 2.0 * np.pi * np.arange(n_theta) / n_theta
    directions = np.column_stack((np.cos(ray_angles), np.sin(ray_angles)))
    outer_distances = distance_to_outer_boundary(ray_angles)
    if np.any(outer_distances <= HOLE_RADIUS):
        raise ValueError("The hole must lie strictly inside the exterior")

    center = np.asarray(HOLE_CENTER)
    point_rings = []
    for radial_index in range(n_radial + 1):
        fraction = radial_index / n_radial
        # Slight grading towards the hole improves its curved-boundary resolution.
        graded_fraction = fraction**1.25
        distances = HOLE_RADIUS + graded_fraction * (
            outer_distances - HOLE_RADIUS
        )
        point_rings.append(center + distances[:, None] * directions)
    points = np.concatenate(point_rings, axis=0)

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
    triangles_array = np.asarray(triangles, dtype=np.int32)

    first = points[triangles_array[:, 0]]
    second = points[triangles_array[:, 1]]
    third = points[triangles_array[:, 2]]
    signed_double_area = (
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    )
    if np.any(signed_double_area <= 0.0):
        raise ValueError(
            "The chosen geometry is not star-shaped about the hole center; "
            "the radial mesh contains an inverted element"
        )
    return MeshTri(points.T, triangles_array.T)


@BilinearForm
def plate_bilinear_form(u, v, _):
    """Kirchhoff bending-energy bilinear form."""
    return FLEXURAL_RIGIDITY * (
        (1.0 - POISSON_RATIO) * ddot(dd(u), dd(v))
        + POISSON_RATIO * trace(dd(u)) * trace(dd(v))
    )


@LinearForm
def transverse_load(v, w):
    """Weak transverse-load functional, identical to the B-spline load."""
    load = (
        LOAD_AMPLITUDE
        * np.sin(np.pi * w.x[0] / PLATE_LENGTH)
        * np.sin(np.pi * w.x[1] / PLATE_LENGTH)
    )
    return load * v


def split_boundary_facets(mesh: MeshTri) -> tuple[np.ndarray, np.ndarray]:
    """Separate outer and hole facets using distance from the hole center."""
    boundary_facets = mesh.boundary_facets()
    facet_vertices = mesh.facets[:, boundary_facets]
    midpoints = mesh.p[:, facet_vertices].mean(axis=1).T
    distance_from_hole = np.linalg.norm(
        midpoints - np.asarray(HOLE_CENTER), axis=1
    )
    # Hole facets are approximately R from the hole center; all exterior
    # facets are much farther away for the selected geometry.
    threshold = 2.0 * HOLE_RADIUS
    hole_mask = distance_from_hole < threshold
    if not np.any(hole_mask) or np.all(hole_mask):
        raise RuntimeError("Could not distinguish outer and hole boundary facets")
    return boundary_facets[~hole_mask], boundary_facets[hole_mask]


def solve_plate(cells_per_direction: int):
    """Assemble and solve the nonconforming Morley plate discretization."""
    mesh = create_mesh(cells_per_direction)
    basis = Basis(mesh, ElementTriMorley())
    stiffness = asm(plate_bilinear_form, basis)
    force = asm(transverse_load, basis)
    outer_facets, _ = split_boundary_facets(mesh)

    # For a simply supported edge, w=0 is essential and constrained here.
    # M_nn=0 is natural. Both M_nn=0 and effective shear=0 are natural on
    # the unconstrained circular-hole boundary.
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
    """Save nodal values and all geometry metadata needed for validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=mesh.p.T,
        triangles=mesh.t.T,
        deflection=nodal_deflection,
        cells_per_direction=np.asarray(cells_per_direction),
        geometry=np.asarray("smooth_polar_irregular"),
        plate_length=np.asarray(PLATE_LENGTH),
        outer_center=np.asarray(OUTER_CENTER),
        outer_mean_radius=np.asarray(OUTER_MEAN_RADIUS),
        outer_cos3_amplitude=np.asarray(OUTER_COS3_AMPLITUDE),
        outer_sin2_amplitude=np.asarray(OUTER_SIN2_AMPLITUDE),
        hole_center=np.asarray(HOLE_CENTER),
        hole_radius=np.asarray(HOLE_RADIUS),
        load_amplitude=np.asarray(LOAD_AMPLITUDE),
        flexural_rigidity=np.asarray(FLEXURAL_RIGIDITY),
        poisson_ratio=np.asarray(POISSON_RATIO),
        outer_boundary=np.asarray("simply_supported"),
        hole_boundary=np.asarray("free"),
    )
    print(f"Saved FEM reference to: {path}")


def plot_reference(mesh: MeshTri, nodal_deflection: np.ndarray) -> None:
    """Plot the body-fitted mesh solution."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.25)
    axes[0].set_title("Body-fitted FEM mesh")
    contour = axes[1].tricontourf(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        nodal_deflection,
        levels=50,
        cmap="viridis",
    )
    figure.colorbar(contour, ax=axes[1], label="FEM deflection w")
    axes[1].set_title("Morley FEM reference")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    plt.show()


def main(cells_per_direction: int, show_plot: bool) -> None:
    print(f"Solving with mesh resolution parameter {cells_per_direction}...")
    mesh, nodal_deflection = solve_plate(cells_per_direction)
    save_reference(mesh, nodal_deflection, cells_per_direction)
    outer_facets, hole_facets = split_boundary_facets(mesh)
    print(f"Mesh vertices: {mesh.nvertices}")
    print(f"Mesh triangles: {mesh.nelements}")
    print(f"Outer boundary facets: {outer_facets.size}")
    print(f"Hole boundary facets: {hole_facets.size}")
    print(f"Maximum deflection: {nodal_deflection.max():.6e}")
    print(f"Minimum deflection: {nodal_deflection.min():.6e}")
    if show_plot:
        plot_reference(mesh, nodal_deflection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Morley FEM reference for the irregular plate with an "
            "off-center free hole"
        )
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=DEFAULT_CELLS_PER_DIRECTION,
        help="mesh resolution parameter",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="save the reference without opening a figure",
    )
    arguments = parser.parse_args()
    main(arguments.cells, not arguments.no_plot)
