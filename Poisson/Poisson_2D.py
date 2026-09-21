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


HYPERPARAMETER_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "best_bspline_hyperparameters.json"
)


### B-spline representation for the 2D Poisson problem ###
class bsplines:
   

    def __init__(self, n_basis, degree):
        self.degree = degree
        self.n_basis = n_basis
        self.n_knots = n_basis + degree + 1
        knot_vector = torch.linspace(0, 1, self.n_knots - 2 * degree)
        self.knot_vector = torch.cat((
            torch.full((degree,), knot_vector[0]), knot_vector,
            torch.full((degree,), knot_vector[-1]),
        ))
        self.knot_x = self.knot_vector.clone()
        self.knot_y = self.knot_vector.clone()
        self.coeffs = torch.rand(n_basis, n_basis, requires_grad=True)

    def basis_matrix(self, x, degree=None, knots=None):
        """Evaluate all B-spline basis functions without autograd."""
        degree = self.degree if degree is None else degree
        knots = self.knot_vector if knots is None else knots
        knots = knots.to(device=x.device, dtype=x.dtype)
        B = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).to(x.dtype)
        for p in range(1, degree + 1):
            n = len(knots) - p - 1
            next_B = torch.zeros((x.numel(), n), device=x.device, dtype=x.dtype)
            for i in range(n):
                denom1 = knots[i + p] - knots[i]
                denom2 = knots[i + p + 1] - knots[i + 1]
                if denom1 > 0:
                    next_B[:, i] += (x - knots[i]) / denom1 * B[:, i]
                if denom2 > 0:
                    next_B[:, i] += (knots[i + p + 1] - x) / denom2 * B[:, i + 1]
            B = next_B
        at_right_endpoint = x == knots[-1]
        if torch.any(at_right_endpoint):
            B[at_right_endpoint] = 0
            B[at_right_endpoint, -1] = 1
        return B

    def basis_and_second_derivative(self, x, knots=None):
        """Return B_{i,p}(x) and its analytic second derivative."""
        knots = self.knot_vector if knots is None else knots
        knots = knots.to(device=x.device, dtype=x.dtype)
        p = self.degree
        if p < 2:
            raise ValueError("Second derivatives require degree >= 2")
        B = self.basis_matrix(x, degree=p, knots=knots)
        B_low = self.basis_matrix(x, degree=p - 2, knots=knots)
        D_low = torch.zeros(
            (x.numel(), self.n_basis + 1), device=x.device, dtype=x.dtype
        )
        q = p - 1
        for i in range(self.n_basis + 1):
            denom1 = knots[i + q] - knots[i]
            denom2 = knots[i + q + 1] - knots[i + 1]
            if denom1 > 0:
                D_low[:, i] += q / denom1 * B_low[:, i]
            if denom2 > 0:
                D_low[:, i] -= q / denom2 * B_low[:, i + 1]
        D2 = torch.zeros_like(B)
        for i in range(self.n_basis):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom1 > 0:
                D2[:, i] += p / denom1 * D_low[:, i]
            if denom2 > 0:
                D2[:, i] -= p / denom2 * D_low[:, i + 1]
        return B, D2

    def forward(self, x):
        B_x = self.basis_matrix(x[:, 0], knots=self.knot_x)[:, 1:-1]
        B_y = self.basis_matrix(x[:, 1], knots=self.knot_y)[:, 1:-1]
        return torch.einsum("ni,ij,nj->n", B_x, self.coeffs[1:-1, 1:-1], B_y)


### Right-hand side of the Poisson equation. ###
def f(x, y):
    
    return -2 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

### Build the fixed Laplacian collocation matrix and source vector based on the collocation points. ###
def build_poisson_system(model, points):
    
    with torch.no_grad():
        Bx, Bxx = model.basis_and_second_derivative(points[:, 0], model.knot_x)
        By, Byy = model.basis_and_second_derivative(points[:, 1], model.knot_y)
        Bx, Bxx = Bx[:, 1:-1], Bxx[:, 1:-1]
        By, Byy = By[:, 1:-1], Byy[:, 1:-1]
        laplacian_matrix = (
            Bxx[:, :, None] * By[:, None, :]
            + Bx[:, :, None] * Byy[:, None, :]
        ).reshape(points.shape[0], -1)
        source = f(points[:, 0], points[:, 1])
    return laplacian_matrix, source

### Load the best B-spline hyperparameters from the tuning results. ###
def load_best_bspline_hyperparameters(path=HYPERPARAMETER_PATH):
    """Load and validate the B-spline configuration saved by Optuna."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"B-spline hyperparameter file not found: {path}. "
            "Run Poisson/tune_bsplines.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    try:
        n_basis = int(result["best_params"]["n_basis"])
        degree = int(result["best_params"]["degree"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid B-spline hyperparameter file: {path}"
        ) from error

    if degree < 2:
        raise ValueError("The B-spline degree must be at least 2")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than the B-spline degree")

    return n_basis, degree


### Main function: train the B-spline model to solve the 2D Poisson problem . ###
def main():
    
    torch.manual_seed(42)

    ### Load the best B-spline hyperparameters from the tuning results. ###
    n_basis, degree = load_best_bspline_hyperparameters()
    print(f"Using tuned B-splines: n_basis={n_basis}, degree={degree}")

    ### Create the B-spline model with the loaded hyperparameters. ###
    model = bsplines(n_basis, degree)

    ### Generate random collocation points in the unit square (full batch training). ###
    X = torch.rand(5000, 2)

    ### Computation of the Laplacian matrix and source vector on collocation points. ###
    laplacian_matrix, source = build_poisson_system(model, X)

    ### Define the loss function based on the Poisson residual. ###
    def poisson_residual_loss():
            interior_coeffs = model.coeffs[1:-1, 1:-1].reshape(-1)
            residual = laplacian_matrix @ interior_coeffs - source
            return residual.square().mean()

    ### Training settings. ###
    epochs = 20000
    learning_rate = 1e-3
    optimizer = torch.optim.Adam([model.coeffs], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, verbose=True, min_lr=1e-5
    )
    early_stopping_patience = 500
    min_delta = 1e-8

    ### Training loop. ###
    loss_vec = train(
        parameters=[model.coeffs], loss_fn=poisson_residual_loss,
        optimizer=optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta, print_every=5,
    )


    #### Visualization of the B-spline solution, exact solution, and absolute error. ###
    x = y = np.linspace(0, 1, 100)
    Xg, Yg = np.meshgrid(x, y)
    XY = np.vstack([Xg.ravel(), Yg.ravel()]).T
    XY_torch = torch.tensor(XY, dtype=torch.float32)
    with torch.no_grad():
        u_pred = model.forward(XY_torch).numpy().reshape(Xg.shape)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    u_exact = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)

    im0 = axs[0].imshow(u_pred, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    axs[0].set_title("PI-TP-BsN Solution", fontsize=14)
    cbar = fig.colorbar(im0, ax=axs[0])
    cbar.ax.tick_params(labelsize=12)
    im1 = axs[1].imshow(u_exact, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    axs[1].set_title("Exact Solution", fontsize=14)
    cbar = fig.colorbar(im1, ax=axs[1])
    cbar.ax.tick_params(labelsize=12)
    error = np.abs(u_pred - u_exact)
    im2 = axs[2].imshow(error, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
    axs[2].set_title("Absolute Error", fontsize=14)
    cbar = fig.colorbar(im2, ax=axs[2])
    cbar.ax.tick_params(labelsize=12)
    for ax in axs:
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()

    ### Visualization of the loss curve. ###
    plt.plot(np.array(loss_vec))
    plt.yscale("log")
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
