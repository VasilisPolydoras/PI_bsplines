import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"


def load_configuration(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"B-spline configuration not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "reproducibility", "runtime", "problem", "model", "sampling",
        "boundary_preprocessing", "training", "tuning", "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


CONFIG = load_configuration()
SEED = int(CONFIG["reproducibility"]["seed"])
REYNOLDS_NUMBER = float(CONFIG["problem"]["reynolds_number"])
X_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["x_bounds"])
Y_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["y_bounds"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_BASIS = int(CONFIG["model"]["n_basis"])
DEGREE = int(CONFIG["model"]["degree"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS = int(CONFIG["sampling"]["boundary_points_per_edge"])
PLOT_X_POINTS = int(CONFIG["sampling"]["plot_x_points"])
PLOT_Y_POINTS = int(CONFIG["sampling"]["plot_y_points"])
BOUNDARY_EPOCHS = int(CONFIG["boundary_preprocessing"]["epochs"])
INTERIOR_EPOCHS = int(CONFIG["training"]["epochs"])
BOUNDARY_LEARNING_RATE = float(CONFIG["boundary_preprocessing"]["learning_rate"])
INTERIOR_LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
BOUNDARY_SCHEDULER_FACTOR = float(
    CONFIG["boundary_preprocessing"]["scheduler"]["factor"]
)
BOUNDARY_SCHEDULER_PATIENCE = int(
    CONFIG["boundary_preprocessing"]["scheduler"]["patience"]
)
BOUNDARY_MIN_LEARNING_RATE = float(
    CONFIG["boundary_preprocessing"]["scheduler"]["min_learning_rate"]
)
BOUNDARY_STOPPING_PATIENCE = int(
    CONFIG["boundary_preprocessing"]["early_stopping"]["patience"]
)
BOUNDARY_MIN_DELTA = float(
    CONFIG["boundary_preprocessing"]["early_stopping"]["min_delta"]
)
BOUNDARY_PRINT_EVERY = int(CONFIG["boundary_preprocessing"]["print_every"])
INTERIOR_SCHEDULER_FACTOR = float(CONFIG["training"]["scheduler"]["factor"])
INTERIOR_SCHEDULER_PATIENCE = int(CONFIG["training"]["scheduler"]["patience"])
INTERIOR_MIN_LEARNING_RATE = float(
    CONFIG["training"]["scheduler"]["min_learning_rate"]
)
INTERIOR_STOPPING_PATIENCE = int(
    CONFIG["training"]["early_stopping"]["patience"]
)
INTERIOR_MIN_DELTA = float(CONFIG["training"]["early_stopping"]["min_delta"])
INTERIOR_PRINT_EVERY = int(CONFIG["training"]["print_every"])
HYPERPARAMETER_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]


class bsplines:
    """Tensor-product B-spline approximation of velocity and pressure."""

    def __init__(self, n_basis, degree, device=None):
        device = torch.device("cpu") if device is None else torch.device(device)
        self.n_basis = n_basis
        self.degree = degree
        self.knot_x = self._clamped_knots(X_BOUNDS, device)
        self.knot_y = self._clamped_knots(Y_BOUNDS, device)
        self.coeffs = torch.rand(
            3, n_basis, n_basis, device=device, requires_grad=True
        )
        self.boundary_coeffs = None
        self.interior_coeffs = None

    def coefficient_tensor(self):
        """Assemble C from frozen boundary edges and trainable interior values."""
        if self.interior_coeffs is None:
            return self.coeffs
        coefficients = self.boundary_coeffs.clone()
        coefficients[:, 1:-1, 1:-1] = self.interior_coeffs
        return coefficients

    def set_boundary_coefficients(self, boundary_coefficients):
        """Freeze the four fitted edges and initialize a new interior block."""
        if boundary_coefficients.shape != self.coeffs.shape:
            raise ValueError(
                f"Expected shape {tuple(self.coeffs.shape)}, "
                f"got {tuple(boundary_coefficients.shape)}"
            )
        self.boundary_coeffs = boundary_coefficients.detach().clone()
        initial_coefficients = self._coons_initialization(self.boundary_coeffs)
        self.interior_coeffs = (
            initial_coefficients[:, 1:-1, 1:-1]
            .detach()
            .clone()
            .requires_grad_(True)
        )

    def _coons_initialization(self, boundary_coefficients):
        """Smoothly extend the four fitted coefficient edges into the domain."""
        dtype = boundary_coefficients.dtype
        device = boundary_coefficients.device
        s = torch.linspace(0, 1, self.n_basis, device=device, dtype=dtype)
        t = torch.linspace(0, 1, self.n_basis, device=device, dtype=dtype)
        sx = s[None, :, None]
        ty = t[None, None, :]

        left = boundary_coefficients[:, 0, :][:, None, :]
        right = boundary_coefficients[:, -1, :][:, None, :]
        bottom = boundary_coefficients[:, :, 0][:, :, None]
        top = boundary_coefficients[:, :, -1][:, :, None]
        corners = (
            (1 - sx) * (1 - ty) * boundary_coefficients[:, 0, 0][:, None, None]
            + (1 - sx) * ty * boundary_coefficients[:, 0, -1][:, None, None]
            + sx * (1 - ty) * boundary_coefficients[:, -1, 0][:, None, None]
            + sx * ty * boundary_coefficients[:, -1, -1][:, None, None]
        )
        coefficients = (
            (1 - sx) * left
            + sx * right
            + (1 - ty) * bottom
            + ty * top
            - corners
        )
        coefficients[:, 0, :] = boundary_coefficients[:, 0, :]
        coefficients[:, -1, :] = boundary_coefficients[:, -1, :]
        coefficients[:, :, 0] = boundary_coefficients[:, :, 0]
        coefficients[:, :, -1] = boundary_coefficients[:, :, -1]
        return coefficients

    def _clamped_knots(self, bounds, device):
        n_knots = self.n_basis + self.degree + 1
        values = torch.linspace(
            bounds[0], bounds[1], n_knots - 2 * self.degree, device=device
        )
        return torch.cat(
            (
                torch.full((self.degree,), values[0], device=device),
                values,
                torch.full((self.degree,), values[-1], device=device),
            )
        )

    def basis_matrix(self, x, knots, degree=None):
        degree = self.degree if degree is None else degree
        knots = knots.to(device=x.device, dtype=x.dtype)
        basis = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).to(x.dtype)
        for p in range(1, degree + 1):
            count = len(knots) - p - 1
            next_basis = torch.zeros(
                (x.numel(), count), device=x.device, dtype=x.dtype
            )
            for i in range(count):
                left = knots[i + p] - knots[i]
                right = knots[i + p + 1] - knots[i + 1]
                if left > 0:
                    next_basis[:, i] += (x - knots[i]) / left * basis[:, i]
                if right > 0:
                    next_basis[:, i] += (
                        (knots[i + p + 1] - x) / right * basis[:, i + 1]
                    )
            basis = next_basis
        at_right = x == knots[-1]
        if torch.any(at_right):
            basis[at_right] = 0
            basis[at_right, -1] = 1
        return basis

    def basis_and_derivatives(self, x, knots):
        p = self.degree
        if p < 2:
            raise ValueError("First and second derivatives require degree >= 2")
        knots = knots.to(device=x.device, dtype=x.dtype)
        basis = self.basis_matrix(x, knots, p)
        lower = self.basis_matrix(x, knots, p - 1)
        first = torch.zeros_like(basis)
        for i in range(self.n_basis):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                first[:, i] += p / left * lower[:, i]
            if right > 0:
                first[:, i] -= p / right * lower[:, i + 1]

        lowest = self.basis_matrix(x, knots, p - 2)
        lower_first = torch.zeros(
            (x.numel(), self.n_basis + 1), device=x.device, dtype=x.dtype
        )
        q = p - 1
        for i in range(self.n_basis + 1):
            left = knots[i + q] - knots[i]
            right = knots[i + q + 1] - knots[i + 1]
            if left > 0:
                lower_first[:, i] += q / left * lowest[:, i]
            if right > 0:
                lower_first[:, i] -= q / right * lowest[:, i + 1]

        second = torch.zeros_like(basis)
        for i in range(self.n_basis):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                second[:, i] += p / left * lower_first[:, i]
            if right > 0:
                second[:, i] -= p / right * lower_first[:, i + 1]
        return basis, first, second

    def forward(self, points):
        bx = self.basis_matrix(points[:, 0], self.knot_x)
        by = self.basis_matrix(points[:, 1], self.knot_y)
        return torch.einsum(
            "ni,kij,nj->nk", bx, self.coefficient_tensor(), by
        )


def exact_solution(points, reynolds_number=REYNOLDS_NUMBER):
    x, y = points[:, 0], points[:, 1]
    lam = reynolds_number / 2 - torch.sqrt(
        torch.as_tensor(
            reynolds_number**2 / 4 + 4 * np.pi**2,
            device=points.device,
            dtype=points.dtype,
        )
    )
    exponential = torch.exp(lam * x)
    u = 1 - exponential * torch.cos(2 * torch.pi * y)
    v = lam / (2 * torch.pi) * exponential * torch.sin(2 * torch.pi * y)
    p = 0.5 * (1 - torch.exp(2 * lam * x))
    return torch.stack((u, v, p), dim=1)


def sample_interior(n_points, device, generator=None):
    unit = torch.rand(n_points, 2, device=device, generator=generator)
    unit[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * unit[:, 0]
    unit[:, 1] = Y_BOUNDS[0] + (Y_BOUNDS[1] - Y_BOUNDS[0]) * unit[:, 1]
    return unit


def sample_boundary(points_per_edge, device):
    x = torch.linspace(*X_BOUNDS, points_per_edge, device=device)
    y = torch.linspace(*Y_BOUNDS, points_per_edge, device=device)
    return torch.cat(
        (
            torch.stack((x, torch.full_like(x, Y_BOUNDS[0])), dim=1),
            torch.stack((x, torch.full_like(x, Y_BOUNDS[1])), dim=1),
            torch.stack((torch.full_like(y, X_BOUNDS[0]), y), dim=1),
            torch.stack((torch.full_like(y, X_BOUNDS[1]), y), dim=1),
        )
    )


def precompute_operators(model, points):
    with torch.no_grad():
        bx, bx_x, bx_xx = model.basis_and_derivatives(points[:, 0], model.knot_x)
        by, by_y, by_yy = model.basis_and_derivatives(points[:, 1], model.knot_y)
    return bx, bx_x, bx_xx, by, by_y, by_yy


def field_values(coefficients, operators):
    bx, bx_x, bx_xx, by, by_y, by_yy = operators
    value = torch.einsum("ni,ij,nj->n", bx, coefficients, by)
    dx = torch.einsum("ni,ij,nj->n", bx_x, coefficients, by)
    dy = torch.einsum("ni,ij,nj->n", bx, coefficients, by_y)
    dxx = torch.einsum("ni,ij,nj->n", bx_xx, coefficients, by)
    dyy = torch.einsum("ni,ij,nj->n", bx, coefficients, by_yy)
    return value, dx, dy, dxx, dyy


def boundary_loss(model, boundary_points, reynolds_number=REYNOLDS_NUMBER):
    error = model.forward(boundary_points) - exact_solution(
        boundary_points, reynolds_number
    )
    return error.square().mean()


def extract_boundary_coefficients(coefficients):
    """Keep the four outer edges of C and clear its unused interior."""
    boundary_coefficients = torch.zeros_like(coefficients)
    boundary_coefficients[:, 0, :] = coefficients[:, 0, :]
    boundary_coefficients[:, -1, :] = coefficients[:, -1, :]
    boundary_coefficients[:, :, 0] = coefficients[:, :, 0]
    boundary_coefficients[:, :, -1] = coefficients[:, :, -1]
    return boundary_coefficients.detach()


def pretrain_boundary_coefficients(
    n_basis,
    degree,
    boundary_points,
    device,
    epochs=BOUNDARY_EPOCHS,
):
    """Optimize the boundary fit in an independent preprocessing stage."""
    boundary_model = bsplines(n_basis, degree, device)
    with torch.no_grad():
        boundary_model.coeffs.zero_()
        bx = boundary_model.basis_matrix(
            boundary_points[:, 0], boundary_model.knot_x
        )
        by = boundary_model.basis_matrix(
            boundary_points[:, 1], boundary_model.knot_y
        )
        targets = exact_solution(boundary_points)

    def loss_fn():
        prediction = torch.einsum(
            "ni,kij,nj->nk", bx, boundary_model.coeffs, by
        )
        return (prediction - targets).square().mean()

    optimizer = torch.optim.Adam(
        [boundary_model.coeffs], lr=BOUNDARY_LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=BOUNDARY_SCHEDULER_FACTOR,
        patience=BOUNDARY_SCHEDULER_PATIENCE,
        min_lr=BOUNDARY_MIN_LEARNING_RATE,
    )
    history = train(
        [boundary_model.coeffs],
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=BOUNDARY_STOPPING_PATIENCE,
        min_delta=BOUNDARY_MIN_DELTA,
        print_every=BOUNDARY_PRINT_EVERY,
    )
    return extract_boundary_coefficients(boundary_model.coeffs), history


def kovasznay_pde_loss(
    model, interior_operators, reynolds_number=REYNOLDS_NUMBER
):
    coefficients = model.coefficient_tensor()
    u, u_x, u_y, u_xx, u_yy = field_values(
        coefficients[0], interior_operators
    )
    v, v_x, v_y, v_xx, v_yy = field_values(
        coefficients[1], interior_operators
    )
    _, p_x, p_y, _, _ = field_values(coefficients[2], interior_operators)
    viscosity = 1.0 / reynolds_number
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy)
    return (
        continuity.square().mean()
        + momentum_x.square().mean()
        + momentum_y.square().mean()
    )


def kovasznay_loss(
    model, interior_operators, boundary_points, reynolds_number=REYNOLDS_NUMBER
):
    """Combined loss retained for the existing Optuna tuning script."""
    return kovasznay_pde_loss(
        model, interior_operators, reynolds_number
    ) + boundary_loss(
        model, boundary_points, reynolds_number
    )


def load_best_hyperparameters(path=HYPERPARAMETER_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file not found: {path}. Run Kovasznay/tune_bsplines.py first."
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    return int(result["best_params"]["n_basis"]), int(result["best_params"]["degree"])


def select_device(name):
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def main(
    device_name=DEFAULT_DEVICE,
    boundary_epochs=BOUNDARY_EPOCHS,
    interior_epochs=INTERIOR_EPOCHS,
    n_basis=N_BASIS,
    degree=DEGREE,
    use_tuned=None,
):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and HYPERPARAMETER_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_basis, degree = load_best_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {HYPERPARAMETER_PATH}")
    print(f"Using device: {device}")
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(f"Using tuned B-splines: n_basis={n_basis}, degree={degree}")
    interior = sample_interior(N_INTERIOR_POINTS, device)
    boundary = sample_boundary(N_BOUNDARY_POINTS, device)

    print("Stage 1/2: preprocessing boundary conditions")
    boundary_coefficients, boundary_history = pretrain_boundary_coefficients(
        n_basis,
        degree,
        boundary,
        device,
        epochs=boundary_epochs,
    )

    # Create a separate PDE model. The boundary edges are copied and detached;
    # its interior is initialized from those edges, then optimized independently.
    model = bsplines(n_basis, degree, device)
    model.set_boundary_coefficients(boundary_coefficients)
    operators = precompute_operators(model, interior)
    initial_boundary_loss = boundary_loss(model, boundary).item()
    print(f"Frozen boundary loss: {initial_boundary_loss:.6e}")

    print("Stage 2/2: optimizing interior coefficients with PDE residual only")
    optimizer = torch.optim.Adam(
        [model.interior_coeffs], lr=INTERIOR_LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=INTERIOR_SCHEDULER_FACTOR,
        patience=INTERIOR_SCHEDULER_PATIENCE,
        min_lr=INTERIOR_MIN_LEARNING_RATE,
    )
    history = train(
        [model.interior_coeffs],
        lambda: kovasznay_pde_loss(model, operators),
        optimizer,
        epochs=interior_epochs,
        scheduler=scheduler,
        early_stopping_patience=INTERIOR_STOPPING_PATIENCE,
        min_delta=INTERIOR_MIN_DELTA,
        print_every=INTERIOR_PRINT_EVERY,
    )
    final_boundary_loss = boundary_loss(model, boundary).item()
    print(f"Boundary loss after PDE training: {final_boundary_loss:.6e}")

    x = np.linspace(*X_BOUNDS, PLOT_X_POINTS)
    y = np.linspace(*Y_BOUNDS, PLOT_Y_POINTS)
    gx, gy = np.meshgrid(x, y)
    points = torch.tensor(
        np.column_stack((gx.ravel(), gy.ravel())), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        prediction = model.forward(points)
        exact = exact_solution(points)
    prediction = prediction.cpu().numpy().reshape(*gx.shape, 3)
    exact = exact.cpu().numpy().reshape(*gx.shape, 3)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    names = ("u", "v", "p")
    for row, name in enumerate(names):
        values = (prediction[:, :, row], exact[:, :, row], np.abs(prediction[:, :, row] - exact[:, :, row]))
        for col, (value, title) in enumerate(zip(values, (f"Predicted {name}", f"Exact {name}", f"Error {name}"))):
            image = axes[row, col].imshow(value, extent=[*X_BOUNDS, *Y_BOUNDS], origin="lower", aspect="auto")
            axes[row, col].set_title(title)
            fig.colorbar(image, ax=axes[row, col])
        relative_error = np.linalg.norm(values[0] - values[1]) / np.linalg.norm(values[1])
        print(f"Relative L2 error {name}: {relative_error:.6e}")
    plt.tight_layout()
    plt.show()

    plt.plot(boundary_history, label="Boundary preprocessing")
    plt.plot(history, label="Interior PDE training")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Kovasznay B-spline Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Kovasznay flow with B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--boundary-epochs", type=int, default=BOUNDARY_EPOCHS)
    parser.add_argument("--interior-epochs", type=int, default=INTERIOR_EPOCHS)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(
        args.device, args.boundary_epochs, args.interior_epochs,
        args.n_basis, args.degree, args.use_tuned,
    )
