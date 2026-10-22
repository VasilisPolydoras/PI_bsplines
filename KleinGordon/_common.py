"""Shared benchmark, constraints, sampling, diagnostics, and plotting helpers."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

X_BOUNDS = (-1.0, 1.0)
Y_BOUNDS = (-1.0, 1.0)
T_BOUNDS = (0.0, 10.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'cpu' or 'gpu'")


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """u=(x+y)cos(2t)+xy sin(2t), used in arXiv:2408.13101."""
    x, y, time = points.unbind(dim=1)
    return (x + y) * torch.cos(2.0 * time) + x * y * torch.sin(2.0 * time)


def initial_value_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def initial_velocity_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2.0 * x * y


def forcing(points: torch.Tensor) -> torch.Tensor:
    """Manufactured forcing f=u_exact**2-4*u_exact."""
    solution = exact_solution(points)
    return solution.square() - 4.0 * solution


def _coons_from_function(points: torch.Tensor, function) -> torch.Tensor:
    """Transfinite interpolation of four prescribed spatial edge traces."""
    x, y, time = points.unbind(dim=1)
    sx = (x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0])
    sy = (y - Y_BOUNDS[0]) / (Y_BOUNDS[1] - Y_BOUNDS[0])
    # A symmetric nonlinear edge blend avoids reproducing this benchmark's
    # bilinear manufactured solution throughout the interior from boundary
    # traces alone.  It remains an exact lifting because h(0)=0 and h(1)=1.
    def edge_blend(value: torch.Tensor) -> torch.Tensor:
        left = value.square()
        right = (1.0 - value).square()
        return left / (left + right)

    hx = edge_blend(sx)
    hy = edge_blend(sy)
    left = function(torch.stack((torch.full_like(x, X_BOUNDS[0]), y, time), 1))
    right = function(torch.stack((torch.full_like(x, X_BOUNDS[1]), y, time), 1))
    bottom = function(torch.stack((x, torch.full_like(y, Y_BOUNDS[0]), time), 1))
    top = function(torch.stack((x, torch.full_like(y, Y_BOUNDS[1]), time), 1))
    c00 = function(torch.stack((torch.full_like(x, X_BOUNDS[0]), torch.full_like(y, Y_BOUNDS[0]), time), 1))
    c10 = function(torch.stack((torch.full_like(x, X_BOUNDS[1]), torch.full_like(y, Y_BOUNDS[0]), time), 1))
    c01 = function(torch.stack((torch.full_like(x, X_BOUNDS[0]), torch.full_like(y, Y_BOUNDS[1]), time), 1))
    c11 = function(torch.stack((torch.full_like(x, X_BOUNDS[1]), torch.full_like(y, Y_BOUNDS[1]), time), 1))
    edges = (1.0 - hx) * left + hx * right + (1.0 - hy) * bottom + hy * top
    corners = (
        (1.0 - hx) * (1.0 - hy) * c00
        + hx * (1.0 - hy) * c10
        + (1.0 - hx) * hy * c01
        + hx * hy * c11
    )
    return edges - corners


def _initial_velocity_function(points: torch.Tensor) -> torch.Tensor:
    return initial_velocity_xy(points[:, 0], points[:, 1])


def hard_constraint_transform(
    points: torch.Tensor,
    raw_solution: torch.Tensor,
    enforce_initial_velocity: bool = False,
) -> torch.Tensor:
    """Enforce all spatial Dirichlet data and both initial conditions exactly.

    Only prescribed boundary/initial data enter the lifting; the exact interior
    solution is never added to the approximation.
    """
    x, y, time = points.unbind(dim=1)
    boundary_lifting = _coons_from_function(points, exact_solution)
    initial_points = torch.stack((x, y, torch.zeros_like(time)), dim=1)
    initial_boundary_lifting = _coons_from_function(initial_points, exact_solution)
    velocity_boundary_lifting = _coons_from_function(
        initial_points, _initial_velocity_function
    )
    initial_correction = initial_value_xy(x, y) - initial_boundary_lifting
    velocity_correction = initial_velocity_xy(x, y) - velocity_boundary_lifting
    sx = (x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0])
    sy = (y - Y_BOUNDS[0]) / (Y_BOUNDS[1] - Y_BOUNDS[0])
    distance = 16.0 * sx * (1.0 - sx) * sy * (1.0 - sy)
    time_fraction = (time - T_BOUNDS[0]) / (T_BOUNDS[1] - T_BOUNDS[0])
    if enforce_initial_velocity:
        return (
            boundary_lifting
            + initial_correction
            + time * velocity_correction
            + time_fraction.square() * distance * raw_solution
        )
    # Paper-faithful form: impose only u(x,y,0), leaving u_t(x,y,0) free.
    return boundary_lifting + initial_correction + time_fraction * distance * raw_solution


def sample_interior(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    points = torch.rand(count, 3, device=device, generator=generator)
    points[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * points[:, 0]
    points[:, 1] = Y_BOUNDS[0] + (Y_BOUNDS[1] - Y_BOUNDS[0]) * points[:, 1]
    points[:, 2] = T_BOUNDS[0] + (T_BOUNDS[1] - T_BOUNDS[0]) * points[:, 2]
    return points


def sample_constraints(
    face_count: int,
    initial_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    initial_xy = torch.rand(initial_count, 2, device=device, generator=generator)
    initial_xy[:, 0] = X_BOUNDS[0] + 2.0 * initial_xy[:, 0]
    initial_xy[:, 1] = Y_BOUNDS[0] + 2.0 * initial_xy[:, 1]
    initial = torch.column_stack((initial_xy, torch.zeros(initial_count, device=device)))
    faces = []
    for axis, value in ((0, X_BOUNDS[0]), (0, X_BOUNDS[1]), (1, Y_BOUNDS[0]), (1, Y_BOUNDS[1])):
        face = sample_interior(face_count, device, generator)
        face[:, axis] = value
        faces.append(face)
    return {"initial": initial, "boundary": torch.cat(faces, dim=0)}


def pde_residual(model, points: torch.Tensor) -> torch.Tensor:
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    first = torch.autograd.grad(
        solution, coordinates, torch.ones_like(solution), create_graph=True
    )[0]
    second = []
    for axis in range(3):
        derivative = torch.autograd.grad(
            first[:, axis], coordinates, torch.ones_like(first[:, axis]),
            create_graph=True,
        )[0][:, axis]
        second.append(derivative)
    u_xx, u_yy, u_tt = second
    return u_tt - u_xx - u_yy + solution.square() - forcing(coordinates)


def loss_components(model, interior, constraints) -> dict[str, torch.Tensor]:
    residual_loss = pde_residual(model, interior).square().mean()
    initial = constraints["initial"].detach().clone().requires_grad_(True)
    initial_prediction = model(initial)
    initial_value_loss = (
        initial_prediction - initial_value_xy(initial[:, 0], initial[:, 1])
    ).square().mean()
    if getattr(model, "enforce_initial_velocity", False):
        initial_derivatives = torch.autograd.grad(
            initial_prediction, initial, torch.ones_like(initial_prediction),
            create_graph=True,
        )[0]
        initial_velocity_loss = (
            initial_derivatives[:, 2]
            - initial_velocity_xy(initial[:, 0], initial[:, 1])
        ).square().mean()
    else:
        initial_velocity_loss = torch.zeros((), device=interior.device)
    boundary = constraints["boundary"]
    boundary_loss = (model(boundary) - exact_solution(boundary)).square().mean()
    return {
        "pde": residual_loss,
        "initial_value": initial_value_loss,
        "initial_velocity": initial_velocity_loss,
        "boundary": boundary_loss,
    }


def physics_loss(model, interior, constraints, weights) -> torch.Tensor:
    if getattr(model, "constraint_mode", "soft") == "hard":
        return weights["pde"] * pde_residual(model, interior).square().mean()
    terms = loss_components(model, interior, constraints)
    return sum(weights[name] * value for name, value in terms.items())


def relative_l2_error(model, points: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        reference = exact_solution(points)
        prediction = model(points)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def constraint_errors(model, constraints) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial = constraints["initial"].detach().clone().requires_grad_(True)
    prediction = model(initial)
    velocity = torch.autograd.grad(
        prediction, initial, torch.ones_like(prediction), create_graph=False
    )[0][:, 2]
    value_error = (prediction - exact_solution(initial)).abs().max().detach()
    velocity_error = (
        velocity - initial_velocity_xy(initial[:, 0], initial[:, 1])
    ).abs().max().detach()
    with torch.no_grad():
        boundary = constraints["boundary"]
        boundary_error = (model(boundary) - exact_solution(boundary)).abs().max()
    return value_error, velocity_error, boundary_error


def save_results(
    model,
    history: list[float],
    relative_error: float,
    postprocessing_dir: Path,
    stem: str,
    title: str,
    spatial_points: int,
    plot_times: list[float],
    metadata: dict[str, object],
    show_plots: bool,
) -> None:
    device = next(iter(model.parameters())).device
    x = np.linspace(*X_BOUNDS, spatial_points)
    y = np.linspace(*Y_BOUNDS, spatial_points)
    times = np.asarray(plot_times, dtype=float)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    predictions, references = [], []
    for time in times:
        points = torch.tensor(
            np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, time))),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            predictions.append(model(points).cpu().numpy().reshape(gx.shape))
            references.append(exact_solution(points).cpu().numpy().reshape(gx.shape))
    prediction = np.stack(predictions)
    reference = np.stack(references)
    absolute_error = np.abs(prediction - reference)

    # Dense values on the three visible faces of the (t, x, y) box.  These
    # arrays reproduce the space-time visualization used in the reference
    # paper without storing or evaluating an entire three-dimensional volume.
    box_time = np.linspace(*T_BOUNDS, spatial_points)

    top_t, top_x = np.meshgrid(box_time, x, indexing="xy")
    top_points = torch.tensor(
        np.column_stack(
            (top_x.ravel(), np.full(top_x.size, Y_BOUNDS[1]), top_t.ravel())
        ),
        dtype=torch.float32,
        device=device,
    )

    side_t, side_y = np.meshgrid(box_time, y, indexing="xy")
    side_points = torch.tensor(
        np.column_stack(
            (np.full(side_y.size, X_BOUNDS[0]), side_y.ravel(), side_t.ravel())
        ),
        dtype=torch.float32,
        device=device,
    )

    end_x, end_y = np.meshgrid(x, y, indexing="xy")
    end_points = torch.tensor(
        np.column_stack(
            (end_x.ravel(), end_y.ravel(), np.full(end_x.size, T_BOUNDS[1]))
        ),
        dtype=torch.float32,
        device=device,
    )

    def evaluate_face(points: torch.Tensor, shape: tuple[int, ...]):
        with torch.no_grad():
            predicted = model(points).cpu().numpy().reshape(shape)
            exact = exact_solution(points).cpu().numpy().reshape(shape)
        return predicted, exact, np.abs(predicted - exact)

    top_values = evaluate_face(top_points, top_x.shape)
    side_values = evaluate_face(side_points, side_y.shape)
    end_values = evaluate_face(end_points, end_x.shape)

    # The paper visualizes the solution as a colored point cloud on a full
    # Cartesian space-time grid (rather than as colored exterior surfaces).
    volume_t, volume_y, volume_x = np.meshgrid(
        box_time, y, x, indexing="ij"
    )
    volume_points = torch.tensor(
        np.column_stack(
            (volume_x.ravel(), volume_y.ravel(), volume_t.ravel())
        ),
        dtype=torch.float32,
        device=device,
    )
    predicted_chunks = []
    exact_chunks = []
    with torch.no_grad():
        for start in range(0, volume_points.shape[0], 32768):
            points_chunk = volume_points[start:start + 32768]
            predicted_chunks.append(model(points_chunk).cpu())
            exact_chunks.append(exact_solution(points_chunk).cpu())
    volume_prediction = torch.cat(predicted_chunks).numpy().reshape(volume_t.shape)
    volume_reference = torch.cat(exact_chunks).numpy().reshape(volume_t.shape)
    volume_error = np.abs(volume_prediction - volume_reference)
    postprocessing_dir.mkdir(parents=True, exist_ok=True)
    solution_data = postprocessing_dir / f"{stem}_solution_data.npz"
    training_data = postprocessing_dir / f"{stem}_training_data.npz"
    payload = {
        "x": x, "y": y, "time": times, "prediction": prediction,
        "reference": reference, "absolute_error": absolute_error,
        "validation_relative_l2_error": np.asarray(relative_error),
        "box_time": box_time,
        "box_top_prediction": top_values[0],
        "box_top_reference": top_values[1],
        "box_top_error": top_values[2],
        "box_side_prediction": side_values[0],
        "box_side_reference": side_values[1],
        "box_side_error": side_values[2],
        "box_end_prediction": end_values[0],
        "box_end_reference": end_values[1],
        "box_end_error": end_values[2],
        "volume_prediction": volume_prediction,
        "volume_reference": volume_reference,
        "volume_error": volume_error,
    }
    payload.update({key: np.asarray(value) for key, value in metadata.items()})
    np.savez_compressed(solution_data, **payload)
    loss = np.asarray(history, dtype=float)
    np.savez_compressed(training_data, epoch=np.arange(1, loss.size + 1), loss=loss)
    plot_saved_solution(solution_data, postprocessing_dir / f"{stem}_solution.png", title)
    plot_saved_training(training_data, postprocessing_dir / f"{stem}_training.png", title)
    print(f"Saved solution data to: {solution_data}")
    print(f"Saved training data to: {training_data}")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_saved_solution(data_path: Path, output_path: Path, title: str) -> None:
    """Plot the colored space-time point cloud used by the reference paper."""
    with np.load(data_path, allow_pickle=False) as data:
        x, y, times = data["x"], data["y"], data["time"]
        prediction, reference, error = data["prediction"], data["reference"], data["absolute_error"]
        if "volume_prediction" in data.files:
            box_time = data["box_time"]
            volumes = (
                data["volume_prediction"],
                data["volume_reference"],
                data["volume_error"],
            )
        else:
            # Legacy archives contain only a few x-y slices. Interpolate those
            # slices in time for visualization. A new solver run stores actual
            # dense model evaluations and therefore needs no interpolation.
            box_time = np.linspace(float(times[0]), float(times[-1]), len(x))

            def interpolate_volume(values: np.ndarray) -> np.ndarray:
                flattened = values.reshape(len(times), -1)
                interpolated = np.stack(
                    [np.interp(box_time, times, flattened[:, index])
                     for index in range(flattened.shape[1])],
                    axis=1,
                )
                return interpolated.reshape(len(box_time), len(y), len(x))

            volume_prediction = interpolate_volume(prediction)
            volume_reference = interpolate_volume(reference)
            volumes = (
                volume_prediction,
                volume_reference,
                np.abs(volume_prediction - volume_reference),
            )

    grid_t, grid_y, grid_x = np.meshgrid(box_time, y, x, indexing="ij")
    figure = plt.figure(figsize=(18, 6), constrained_layout=True)

    panels = (
        (volumes[0], "Prediction", "viridis"),
        (volumes[1], "Exact solution", "viridis"),
        (volumes[2], "Absolute error", "magma"),
    )
    for panel, (values, label, color_map) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        if np.isclose(value_min, value_max):
            value_max = value_min + 1.0
        normalization = colors.Normalize(vmin=value_min, vmax=value_max)
        colormap = plt.get_cmap(color_map)
        point_cloud = axis.scatter(
            grid_t.ravel(),
            grid_x.ravel(),
            grid_y.ravel(),
            c=values.ravel(),
            s=0.5,
            cmap=colormap,
            norm=normalization,
            linewidths=0.0,
            rasterized=True,
        )
        axis.set_xlim(float(box_time[0]), float(box_time[-1]))
        axis.set_ylim(float(x[0]), float(x[-1]))
        axis.set_zlim(float(y[0]), float(y[-1]))
        axis.set_xlabel("t")
        axis.set_ylabel("x")
        axis.set_zlabel("y")
        axis.set_title(label)
        figure.colorbar(point_cloud, ax=axis, shrink=0.62, pad=0.08)
    figure.suptitle(f"{title}: space-time solution")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_saved_training(data_path: Path, output_path: Path, title: str) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        epochs, loss = data["epoch"], data["loss"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Physics-informed loss")
    axis.set_title(f"{title} training")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
