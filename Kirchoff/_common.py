"""Shared Kirchhoff-plate physics, training, diagnostics, and plotting."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training import train


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def problem_values(config: dict) -> dict[str, float]:
    problem = config["problem"]
    length_x = float(problem["length_x"])
    length_y = float(problem["length_y"])
    young = float(problem["youngs_modulus"])
    thickness = float(problem["thickness"])
    poisson = float(problem["poisson_ratio"])
    rigidity = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    return {
        "length_x": length_x,
        "length_y": length_y,
        "load_amplitude": float(problem["load_amplitude"]),
        "poisson_ratio": poisson,
        "rigidity": rigidity,
    }


def transverse_load(points: torch.Tensor, values: dict[str, float]) -> torch.Tensor:
    return values["load_amplitude"] * torch.sin(
        math.pi * points[:, 0] / values["length_x"]
    ) * torch.sin(math.pi * points[:, 1] / values["length_y"])


def exact_solution(points: torch.Tensor, values: dict[str, float]) -> torch.Tensor:
    wave_number_squared = (
        (math.pi / values["length_x"]) ** 2
        + (math.pi / values["length_y"]) ** 2
    )
    amplitude = values["load_amplitude"] / (
        values["rigidity"] * wave_number_squared**2
    )
    return amplitude * torch.sin(
        math.pi * points[:, 0] / values["length_x"]
    ) * torch.sin(math.pi * points[:, 1] / values["length_y"])


def sample_interior(
    count: int, values: dict[str, float], device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] *= values["length_x"]
    points[:, 1] *= values["length_y"]
    return points


def sample_boundary(
    count: int, values: dict[str, float], device: torch.device,
) -> dict[str, torch.Tensor]:
    x = torch.linspace(0.0, values["length_x"], count, device=device)
    y = torch.linspace(0.0, values["length_y"], count, device=device)
    return {
        "left": torch.stack((torch.zeros_like(y), y), dim=1),
        "right": torch.stack((torch.full_like(y, values["length_x"]), y), dim=1),
        "bottom": torch.stack((x, torch.zeros_like(x)), dim=1),
        "top": torch.stack((x, torch.full_like(x, values["length_y"])), dim=1),
    }


def derivative(
    output: torch.Tensor, coordinates: torch.Tensor, axis: int, order: int,
) -> torch.Tensor:
    result = output
    for _ in range(order):
        result = torch.autograd.grad(
            result, coordinates, torch.ones_like(result), create_graph=True
        )[0][:, axis]
    return result


def plate_residual(model, points: torch.Tensor, values: dict[str, float]) -> torch.Tensor:
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    w_xxxx = derivative(solution, coordinates, 0, 4)
    w_yyyy = derivative(solution, coordinates, 1, 4)
    w_xx = derivative(solution, coordinates, 0, 2)
    w_xxyy = derivative(w_xx, coordinates, 1, 2)
    return (
        values["rigidity"] * (w_xxxx + 2.0 * w_xxyy + w_yyyy)
        - transverse_load(coordinates, values)
    ) / values["load_amplitude"]


def boundary_terms(model, boundaries, values) -> tuple[torch.Tensor, torch.Tensor]:
    def edge_terms(points: torch.Tensor, vertical: bool):
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(coordinates)
        w_xx = derivative(solution, coordinates, 0, 2)
        w_yy = derivative(solution, coordinates, 1, 2)
        curvature = (
            w_xx + values["poisson_ratio"] * w_yy
            if vertical else
            w_yy + values["poisson_ratio"] * w_xx
        )
        return solution, curvature

    edge_data = [
        edge_terms(boundaries["left"], True),
        edge_terms(boundaries["right"], True),
        edge_terms(boundaries["bottom"], False),
        edge_terms(boundaries["top"], False),
    ]
    deflection = torch.stack([item[0].square().mean() for item in edge_data]).mean()
    center = torch.tensor(
        [[values["length_x"] / 2.0, values["length_y"] / 2.0]],
        device=next(model.parameters()).device,
    )
    scale = math.sqrt(values["length_x"] * values["length_y"]) ** 2 / exact_solution(
        center, values
    )[0]
    moment = torch.stack([(scale * item[1]).square().mean() for item in edge_data]).mean()
    return deflection, moment


def physics_loss(model, interior, boundaries, values, weights) -> torch.Tensor:
    pde = plate_residual(model, interior, values).square().mean()
    if model.constraint_mode == "hard":
        return float(weights["pde"]) * pde
    deflection, moment = boundary_terms(model, boundaries, values)
    return (
        float(weights["pde"]) * pde
        + float(weights["deflection"]) * deflection
        + float(weights["moment"]) * moment
    )


def relative_l2(model, points, values) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points, values)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def constraint_errors(model, boundaries, values) -> tuple[float, float]:
    deflection, moment = boundary_terms(model, boundaries, values)
    return math.sqrt(float(deflection.detach())), math.sqrt(float(moment.detach()))


def train_model(model, loss_fn, training: dict) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler"]["factor"]),
        patience=int(training["scheduler"]["patience"]),
        min_lr=float(training["scheduler"]["min_learning_rate"]),
    )
    return train(
        model.parameters(), loss_fn, optimizer,
        epochs=int(training["epochs"]), scheduler=scheduler,
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=int(training["print_every"]),
    )


def save_results(model, history, error, config, values, method: str) -> None:
    output = config["output"]
    post = Path(__file__).resolve().parent / output["postprocessing_directory"]
    post.mkdir(parents=True, exist_ok=True)
    resolution = int(config["sampling"]["plot_points_per_axis"])
    x = np.linspace(0.0, values["length_x"], resolution)
    y = np.linspace(0.0, values["length_y"], resolution)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    device = next(model.parameters()).device
    points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(grid_x.shape)
        reference = exact_solution(points, values).cpu().numpy().reshape(grid_x.shape)
    absolute_error = np.abs(prediction - reference)
    stem = f"{method}_kirchhoff_{model.constraint_mode}"
    solution_path = post / f"{stem}_solution_data.npz"
    training_path = post / f"{stem}_training_data.npz"
    np.savez_compressed(
        solution_path, x=x, y=y, prediction=prediction, reference=reference,
        absolute_error=absolute_error,
        validation_relative_l2_error=np.asarray(error),
        constraint_mode=np.asarray(model.constraint_mode),
    )
    np.savez_compressed(
        training_path, epoch=np.arange(1, len(history) + 1), loss=np.asarray(history)
    )
    plot_solution(solution_path, post / f"{stem}_solution.png", f"{method.upper()} ({model.constraint_mode})")
    plot_training(training_path, post / f"{stem}_training.png", f"{method.upper()} ({model.constraint_mode})")
    print(f"Saved solution data to: {solution_path}")
    if bool(output["show_plots"]):
        plt.show()
    else:
        plt.close("all")


def plot_solution(data_path: Path, output_path: Path, title: str) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        fields = (data["prediction"], data["reference"], data["absolute_error"])
    gx, gy = np.meshgrid(x, y, indexing="xy")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, field, label in zip(axes, fields, ("Prediction", "Exact", "Absolute error")):
        image = axis.contourf(gx, gy, field, levels=40, cmap="viridis")
        figure.colorbar(image, ax=axis)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(label)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_training(data_path: Path, output_path: Path, title: str) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        epoch, loss = data["epoch"], data["loss"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epoch, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Physics-informed loss")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")

