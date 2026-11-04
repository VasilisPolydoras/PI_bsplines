"""Tensor-product B-spline solver for the (2+1)-D Klein--Gordon PDE.

In hard mode, the inhomogeneous Dirichlet, initial-value, and initial-velocity
coefficient surfaces are pretrained sequentially and frozen before PDE
training. PDE derivatives are evaluated analytically from B-spline matrices.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train

X_BOUNDS, Y_BOUNDS, T_BOUNDS = (-1.0, 1.0), (-1.0, 1.0), (0.0, 10.0)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name == "cpu": return torch.device("cpu")
    if name == "gpu" and torch.cuda.is_available(): return torch.device("cuda")
    if name == "gpu": raise RuntimeError("GPU requested, but CUDA is not available")
    raise ValueError("device must be 'cpu' or 'gpu'")


def exact_solution(points):
    x, y, time = points.unbind(1)
    return (x + y) * torch.cos(2.0 * time) + x * y * torch.sin(2.0 * time)


def initial_value_xy(x, y): return x + y
def initial_velocity_xy(x, y): return 2.0 * x * y


def forcing(points):
    value = exact_solution(points)
    return value.square() - 4.0 * value


def sample_interior(count, device, generator=None):
    points = torch.rand(count, 3, device=device, generator=generator)
    points[:, 0] = -1.0 + 2.0 * points[:, 0]
    points[:, 1] = -1.0 + 2.0 * points[:, 1]
    points[:, 2] = 10.0 * points[:, 2]
    return points


def sample_constraints(face_count, initial_count, device, generator=None):
    initial_xy = -1.0 + 2.0 * torch.rand(initial_count, 2, device=device, generator=generator)
    initial = torch.column_stack((initial_xy, torch.zeros(initial_count, device=device)))
    faces = []
    for axis, value in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)):
        face = sample_interior(face_count, device, generator); face[:, axis] = value; faces.append(face)
    return {"initial": initial, "boundary": torch.cat(faces)}


def relative_l2_error(model, points):
    with torch.no_grad():
        reference, prediction = exact_solution(points), model(points)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def save_results(model, history, relative_error, validation, wall_time, output, stem,
                 title, spatial_points, plot_times, metadata, show_plots,
                 condition_history=None):
    device = next(model.parameters()).device
    x = np.linspace(*X_BOUNDS, spatial_points); y = np.linspace(*Y_BOUNDS, spatial_points)
    times = np.asarray(plot_times); gx, gy = np.meshgrid(x, y, indexing="xy")
    prediction, reference = [], []
    with torch.no_grad():
        for value in times:
            points = torch.tensor(np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, value))), dtype=torch.float32, device=device)
            prediction.append(model(points).cpu().numpy().reshape(gx.shape))
            reference.append(exact_solution(points).cpu().numpy().reshape(gx.shape))
    prediction, reference = np.stack(prediction), np.stack(reference)
    box_time = np.linspace(*T_BOUNDS, spatial_points)
    volume_t, volume_y, volume_x = np.meshgrid(box_time, y, x, indexing="ij")
    volume_points = torch.tensor(np.column_stack((volume_x.ravel(), volume_y.ravel(), volume_t.ravel())), dtype=torch.float32, device=device)
    predicted_chunks, reference_chunks = [], []
    with torch.no_grad():
        for start in range(0, volume_points.shape[0], 32768):
            chunk = volume_points[start:start + 32768]
            predicted_chunks.append(model(chunk).cpu()); reference_chunks.append(exact_solution(chunk).cpu())
        validation_prediction = model(validation).cpu().numpy()
        validation_reference = exact_solution(validation).cpu().numpy()
    volume_prediction = torch.cat(predicted_chunks).numpy().reshape(volume_t.shape)
    volume_reference = torch.cat(reference_chunks).numpy().reshape(volume_t.shape)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(x=x, y=y, time=times, prediction=prediction, reference=reference,
                   absolute_error=np.abs(prediction-reference), box_time=box_time,
                   volume_prediction=volume_prediction, volume_reference=volume_reference,
                   volume_error=np.abs(volume_prediction-volume_reference),
                   validation_points=validation.cpu().numpy(), validation_prediction=validation_prediction,
                   validation_reference=validation_reference, validation_relative_l2_error=relative_error,
                   training_wall_time_seconds=wall_time)
    payload.update({key: np.asarray(value) for key, value in metadata.items()})
    np.savez_compressed(output / f"{stem}_solution_data.npz", **payload)
    loss = np.asarray(history)
    training_payload = dict(
        epoch=np.arange(1, loss.size + 1),
        loss=loss,
        interior_epoch=np.arange(1, loss.size + 1),
        interior_loss=loss,
        training_wall_time_seconds=wall_time,
    )
    for name, values in (condition_history or {}).items():
        values = np.asarray(values)
        training_payload[f"{name}_epoch"] = np.arange(1, values.size + 1)
        training_payload[f"{name}_loss"] = values
    np.savez_compressed(
        output / f"{stem}_training_data.npz", **training_payload
    )
    print(f"Saved solution data to: {output / f'{stem}_solution_data.npz'}")
    print(f"Saved training data to: {output / f'{stem}_training_data.npz'}")

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_BASIS_SPACE = int(CONFIG["model"]["n_basis_space"])
N_BASIS_TIME = int(CONFIG["model"]["n_basis_time"])
DEGREE = int(CONFIG["model"]["degree"])
CONSTRAINT_MODE = str(CONFIG["model"].get("constraint_mode", "hard"))
ENFORCE_INITIAL_VELOCITY = bool(CONFIG["problem"].get("enforce_initial_velocity", False))
SAMPLING = CONFIG["sampling"]
TRAINING = CONFIG["training"]
LOSS_WEIGHTS = {key: float(value) for key, value in CONFIG["loss_weights"].items()}
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]


def clamped_uniform_knots(bounds, n_basis, degree, device):
    """Create an open, clamped, uniform knot vector."""
    if degree < 2 or n_basis <= degree:
        raise ValueError("Require degree >= 2 and n_basis > degree")
    count = n_basis - degree - 1
    internal = torch.linspace(bounds[0], bounds[1], count + 2, device=device)[1:-1]
    return torch.cat((
        torch.full((degree + 1,), bounds[0], device=device), internal,
        torch.full((degree + 1,), bounds[1], device=device),
    ))


def basis_matrix(values, knots, degree):
    """Evaluate all degree-p B-spline basis functions."""
    knots = knots.to(device=values.device, dtype=values.dtype)
    basis = ((values[:, None] >= knots[:-1]) & (values[:, None] < knots[1:])).to(values.dtype)
    for order in range(1, degree + 1):
        count = len(knots) - order - 1
        next_basis = torch.zeros(values.numel(), count, device=values.device, dtype=values.dtype)
        for index in range(count):
            left = knots[index + order] - knots[index]
            right = knots[index + order + 1] - knots[index + 1]
            if left > 0:
                next_basis[:, index] += (values - knots[index]) / left * basis[:, index]
            if right > 0:
                next_basis[:, index] += (knots[index + order + 1] - values) / right * basis[:, index + 1]
        basis = next_basis
    at_right = values == knots[-1]
    if torch.any(at_right):
        basis[at_right] = 0.0
        basis[at_right, -1] = 1.0
    return basis


def basis_and_derivatives(values, knots, degree):
    """Return B, dB/dx and d2B/dx2 using exact spline identities."""
    if degree < 2:
        raise ValueError("Second-order PDE derivatives require degree >= 2")
    knots = knots.to(device=values.device, dtype=values.dtype)
    n_basis = len(knots) - degree - 1
    basis = basis_matrix(values, knots, degree)
    lower = basis_matrix(values, knots, degree - 1)
    first = torch.zeros_like(basis)
    for index in range(n_basis):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if left > 0:
            first[:, index] += degree / left * lower[:, index]
        if right > 0:
            first[:, index] -= degree / right * lower[:, index + 1]

    lowest = basis_matrix(values, knots, degree - 2)
    lower_first = torch.zeros(values.numel(), n_basis + 1, device=values.device, dtype=values.dtype)
    lower_degree = degree - 1
    for index in range(n_basis + 1):
        left = knots[index + lower_degree] - knots[index]
        right = knots[index + lower_degree + 1] - knots[index + 1]
        if left > 0:
            lower_first[:, index] += lower_degree / left * lowest[:, index]
        if right > 0:
            lower_first[:, index] -= lower_degree / right * lowest[:, index + 1]

    second = torch.zeros_like(basis)
    for index in range(n_basis):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if left > 0:
            second[:, index] += degree / left * lower_first[:, index]
        if right > 0:
            second[:, index] -= degree / right * lower_first[:, index + 1]
    return basis, first, second


class KleinGordonBSpline(torch.nn.Module):
    """Tensor spline with optionally pretrained and frozen condition traces."""

    def __init__(self, n_basis_space, n_basis_time, degree, constraint_mode,
                 device, enforce_initial_velocity=ENFORCE_INITIAL_VELOCITY):
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_basis_space = n_basis_space
        self.n_basis_time = n_basis_time
        self.degree = degree
        self.constraint_mode = constraint_mode
        self.enforce_initial_velocity = enforce_initial_velocity
        self.register_buffer("knot_x", clamped_uniform_knots(X_BOUNDS, n_basis_space, degree, device))
        self.register_buffer("knot_y", clamped_uniform_knots(Y_BOUNDS, n_basis_space, degree, device))
        self.register_buffer("knot_t", clamped_uniform_knots(T_BOUNDS, n_basis_time, degree, device))

        shape = (n_basis_space, n_basis_space, n_basis_time)
        fixed = torch.zeros(shape, device=device)
        boundary_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        if constraint_mode == "hard":
            boundary_mask[0, :, :] = True
            boundary_mask[-1, :, :] = True
            boundary_mask[:, 0, :] = True
            boundary_mask[:, -1, :] = True
        initial_mask = torch.zeros_like(boundary_mask)
        if constraint_mode == "hard":
            initial_mask[:, :, 0] = True
            initial_mask &= ~boundary_mask
        velocity_mask = torch.zeros_like(boundary_mask)
        if constraint_mode == "hard" and enforce_initial_velocity:
            velocity_mask[:, :, 1] = True
            velocity_mask &= ~(boundary_mask | initial_mask)
        fixed_mask = boundary_mask | initial_mask | velocity_mask
        boundary_indices = boundary_mask.flatten().nonzero(
            as_tuple=False
        ).squeeze(1)
        initial_indices = initial_mask.flatten().nonzero(
            as_tuple=False
        ).squeeze(1)
        velocity_indices = velocity_mask.flatten().nonzero(
            as_tuple=False
        ).squeeze(1)
        free_indices = (~fixed_mask).flatten().nonzero(as_tuple=False).squeeze(1)
        self.register_buffer("fixed_coefficients", fixed)
        self.register_buffer("fixed_mask", fixed_mask)
        self.register_buffer("boundary_indices", boundary_indices)
        self.register_buffer("initial_indices", initial_indices)
        self.register_buffer("velocity_indices", velocity_indices)
        self.register_buffer("free_indices", free_indices)
        self.boundary_coefficients = torch.nn.Parameter(
            torch.zeros(boundary_indices.numel(), device=device),
            requires_grad=constraint_mode == "hard",
        )
        self.initial_coefficients = torch.nn.Parameter(
            torch.zeros(initial_indices.numel(), device=device),
            requires_grad=constraint_mode == "hard",
        )
        self.velocity_coefficients = torch.nn.Parameter(
            torch.zeros(velocity_indices.numel(), device=device),
            requires_grad=(
                constraint_mode == "hard" and enforce_initial_velocity
            ),
        )
        self.interior_coefficients = torch.nn.Parameter(
            0.01 * torch.randn(free_indices.numel(), device=device)
        )

    def staged_coefficient_tensor(self, indices, values):
        """Insert the active preprocessing variables into the fixed tensor."""
        flat = self.fixed_coefficients.flatten()
        flat = flat.scatter(0, indices, values)
        return flat.reshape(self.fixed_coefficients.shape)

    def freeze_stage(self, indices, values):
        """Copy one pretrained coefficient stage into the fixed tensor."""
        with torch.no_grad():
            self.fixed_coefficients.copy_(
                self.staged_coefficient_tensor(indices, values)
            )
        values.requires_grad_(False)

    def coefficient_tensor(self):
        """Insert trainable interior values into the frozen tensor."""
        flat = self.fixed_coefficients.flatten()
        flat = flat.scatter(0, self.free_indices, self.interior_coefficients)
        return flat.reshape(self.fixed_coefficients.shape)

    def forward(self, points):
        bx = basis_matrix(points[:, 0], self.knot_x, self.degree)
        by = basis_matrix(points[:, 1], self.knot_y, self.degree)
        bt = basis_matrix(points[:, 2], self.knot_t, self.degree)
        return torch.einsum("ni,nj,nk,ijk->n", bx, by, bt, self.coefficient_tensor())


def precompute_operators(model, points):
    """Precompute all basis and derivative matrices at collocation points."""
    with torch.no_grad():
        bx, bx_x, bx_xx = basis_and_derivatives(points[:, 0], model.knot_x, model.degree)
        by, by_y, by_yy = basis_and_derivatives(points[:, 1], model.knot_y, model.degree)
        bt, bt_t, bt_tt = basis_and_derivatives(points[:, 2], model.knot_t, model.degree)
    return bx, bx_x, bx_xx, by, by_y, by_yy, bt, bt_t, bt_tt


def field_values(coefficients, operators):
    """Evaluate u and its analytical derivatives by tensor contraction."""
    bx, bx_x, bx_xx, by, by_y, by_yy, bt, bt_t, bt_tt = operators
    def contract(ax, ay, at):
        return torch.einsum("ni,nj,nk,ijk->n", ax, ay, at, coefficients)
    return (
        contract(bx, by, bt), contract(bx_x, by, bt),
        contract(bx, by_y, bt), contract(bx, by, bt_t),
        contract(bx_xx, by, bt), contract(bx, by_yy, bt),
        contract(bx, by, bt_tt),
    )


def bspline_residual(model, operators, points):
    value, _, _, _, dxx, dyy, dtt = field_values(model.coefficient_tensor(), operators)
    return dtt - dxx - dyy + value.square() - forcing(points)


def bspline_loss(model, interior_operators, interior_points,
                 constraint_operators=None, constraints=None,
                 weights=LOSS_WEIGHTS):
    pde = bspline_residual(model, interior_operators, interior_points).square().mean()
    if model.constraint_mode == "hard":
        return weights["pde"] * pde
    if constraint_operators is None or constraints is None:
        raise ValueError("Soft constraints require operators and points")
    coefficients = model.coefficient_tensor()
    initial = field_values(coefficients, constraint_operators["initial"])
    initial_points = constraints["initial"]
    value_loss = (initial[0] - initial_value_xy(initial_points[:, 0], initial_points[:, 1])).square().mean()
    if model.enforce_initial_velocity:
        velocity_loss = (initial[3] - initial_velocity_xy(initial_points[:, 0], initial_points[:, 1])).square().mean()
    else:
        velocity_loss = torch.zeros((), device=interior_points.device)
    boundary = field_values(coefficients, constraint_operators["boundary"])[0]
    boundary_loss = (boundary - exact_solution(constraints["boundary"])).square().mean()
    return (weights["pde"] * pde + weights["initial_value"] * value_loss
            + weights["initial_velocity"] * velocity_loss
            + weights["boundary"] * boundary_loss)


def boundary_pretraining_loss(model, operators, points):
    coefficients = model.staged_coefficient_tensor(
        model.boundary_indices, model.boundary_coefficients
    )
    prediction = field_values(coefficients, operators)[0]
    return (prediction - exact_solution(points)).square().mean()


def initial_value_pretraining_loss(model, operators, points):
    coefficients = model.staged_coefficient_tensor(
        model.initial_indices, model.initial_coefficients
    )
    prediction = field_values(coefficients, operators)[0]
    target = initial_value_xy(points[:, 0], points[:, 1])
    return (prediction - target).square().mean()


def initial_velocity_pretraining_loss(model, operators, points):
    coefficients = model.staged_coefficient_tensor(
        model.velocity_indices, model.velocity_coefficients
    )
    prediction = field_values(coefficients, operators)[3]
    target = initial_velocity_xy(points[:, 0], points[:, 1])
    return (prediction - target).square().mean()


def _train_condition_stage(
    parameters,
    loss_function,
    *,
    epochs,
    learning_rate,
    scheduler_config,
    stopping_config,
    print_every,
):
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
    )
    return train(
        [parameters],
        loss_function,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=int(stopping_config["patience"]),
        min_delta=float(stopping_config["min_delta"]),
        print_every=print_every,
    )


def pretrain_condition_coefficients(
    model,
    constraint_operators,
    constraints,
    *,
    epochs,
    learning_rate,
    scheduler_config,
    stopping_config,
    print_every,
):
    """Sequentially fit the BC, initial-value and initial-velocity surfaces."""
    if model.constraint_mode != "hard":
        return {}
    histories = {}
    histories["boundary"] = _train_condition_stage(
        model.boundary_coefficients,
        lambda: boundary_pretraining_loss(
            model,
            constraint_operators["boundary"],
            constraints["boundary"],
        ),
        epochs=epochs,
        learning_rate=learning_rate,
        scheduler_config=scheduler_config,
        stopping_config=stopping_config,
        print_every=print_every,
    )
    model.freeze_stage(
        model.boundary_indices, model.boundary_coefficients
    )
    histories["initial_value"] = _train_condition_stage(
        model.initial_coefficients,
        lambda: initial_value_pretraining_loss(
            model,
            constraint_operators["initial"],
            constraints["initial"],
        ),
        epochs=epochs,
        learning_rate=learning_rate,
        scheduler_config=scheduler_config,
        stopping_config=stopping_config,
        print_every=print_every,
    )
    model.freeze_stage(model.initial_indices, model.initial_coefficients)
    if model.enforce_initial_velocity:
        histories["initial_velocity"] = _train_condition_stage(
            model.velocity_coefficients,
            lambda: initial_velocity_pretraining_loss(
                model,
                constraint_operators["initial"],
                constraints["initial"],
            ),
            epochs=epochs,
            learning_rate=learning_rate,
            scheduler_config=scheduler_config,
            stopping_config=stopping_config,
            print_every=print_every,
        )
        model.freeze_stage(
            model.velocity_indices, model.velocity_coefficients
        )
    return histories


def bspline_constraint_errors(model, constraints, constraint_operators):
    with torch.no_grad():
        coefficients = model.coefficient_tensor()
        initial = field_values(coefficients, constraint_operators["initial"])
        initial_points = constraints["initial"]
        value_error = (initial[0] - initial_value_xy(initial_points[:, 0], initial_points[:, 1])).abs().max()
        velocity_error = (initial[3] - initial_velocity_xy(initial_points[:, 0], initial_points[:, 1])).abs().max()
        boundary = field_values(coefficients, constraint_operators["boundary"])[0]
        boundary_error = (boundary - exact_solution(constraints["boundary"])).abs().max()
    return value_error, velocity_error, boundary_error


def load_hyperparameters(path=RESULT_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    if not params:
        raise ValueError(f"No tuned B-spline parameters in {path}")
    return int(params["n_basis_space"]), int(params["n_basis_time"]), int(params["degree"])


def main(device_name=DEFAULT_DEVICE, n_basis_space=N_BASIS_SPACE,
         n_basis_time=N_BASIS_TIME, degree=DEGREE,
         constraint_mode=CONSTRAINT_MODE, epochs=int(TRAINING["epochs"]),
         use_tuned=None, initialization_seed=SEED, sampling_seed=SEED,
         validation_seed=None, save_outputs=True, verbose=True):
    set_seed(initialization_seed)
    device = select_device(device_name)
    should_load = (LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
                   and RESULT_PATH.stat().st_size > 3
                   if use_tuned is None else use_tuned)
    if should_load:
        n_basis_space, n_basis_time, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Klein-Gordon B-spline: n_space={n_basis_space}, n_time={n_basis_time}, degree={degree}, constraints={constraint_mode}")
    model = KleinGordonBSpline(n_basis_space, n_basis_time, degree, constraint_mode, device).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(int(SAMPLING["interior_points"]), device, generator)
    constraints = sample_constraints(int(SAMPLING["boundary_points_per_face"]), int(SAMPLING["initial_points"]), device, generator)
    interior_operators = precompute_operators(model, interior)
    constraint_operators = {name: precompute_operators(model, points) for name, points in constraints.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    condition_history = {}
    if constraint_mode == "hard":
        print(
            "Stage 1/2: sequentially pretraining BC, initial-value, "
            "and initial-velocity coefficients"
        )
        condition_history = pretrain_condition_coefficients(
            model,
            constraint_operators,
            constraints,
            epochs=epochs,
            learning_rate=float(TRAINING["learning_rate"]),
            scheduler_config=TRAINING["scheduler"],
            stopping_config=TRAINING["early_stopping"],
            print_every=int(TRAINING["print_every"]),
        )
    before = bspline_constraint_errors(model, constraints, constraint_operators)
    print(f"Frozen-condition errors before PDE training: initial={before[0].item():.3e}, boundary={before[2].item():.3e}")

    print("Stage 2/2: optimizing the PDE-interior coefficients")
    pde_parameters = (
        [model.interior_coefficients]
        if constraint_mode == "hard"
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(pde_parameters, lr=float(TRAINING["learning_rate"]))
    schedule = TRAINING["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(schedule["factor"]),
        patience=int(schedule["patience"]), min_lr=float(schedule["min_learning_rate"]),
    )
    stopping = TRAINING["early_stopping"]
    history = train(
        pde_parameters,
        lambda: bspline_loss(model, interior_operators, interior, constraint_operators, constraints),
        optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]),
        print_every=int(TRAINING["print_every"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(int(SAMPLING["validation_points"]), device, validation_generator)
    validation_operators = precompute_operators(model, validation)
    relative_error = relative_l2_error(model, validation)
    residual_mse = bspline_residual(model, validation_operators, validation).detach().square().mean()
    value_error, velocity_error, boundary_error = bspline_constraint_errors(model, constraints, constraint_operators)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Training wall time: {wall_time:.3f} s")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-value error: {value_error.item():.6e}")
    velocity_label = "Maximum initial-velocity error" if ENFORCE_INITIAL_VELOCITY else "Initial-velocity diagnostic (not enforced)"
    print(f"{velocity_label}: {velocity_error.item():.6e}")
    print(f"Maximum frozen-boundary error: {boundary_error.item():.6e}")
    result = {"validation_relative_l2_error": relative_error.item(), "training_time_seconds": wall_time}
    if not save_outputs:
        return result
    save_results(
        model, history, relative_error.item(), validation, wall_time, POSTPROCESSING_DIR,
        f"bspline_klein_gordon_{constraint_mode}",
        f"Klein-Gordon B-spline ({constraint_mode})",
        int(SAMPLING["plot_spatial_points"]), list(SAMPLING["plot_times"]),
        {"constraint_mode": constraint_mode, "n_basis_space": n_basis_space,
         "n_basis_time": n_basis_time, "degree": degree},
        bool(CONFIG["output"]["show_plots"]),
        condition_history,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve (2+1)-D Klein-Gordon with tensor B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis-space", type=int, default=N_BASIS_SPACE)
    parser.add_argument("--n-basis-time", type=int, default=N_BASIS_TIME)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.n_basis_space, args.n_basis_time, args.degree,
         args.constraint, args.epochs, args.use_tuned)
