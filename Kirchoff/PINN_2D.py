"""Physics-informed neural network for a simply supported Kirchhoff plate."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SCRIPT_DIR / "pinn_config.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
P = CONFIG["problem"]
LENGTH_X, LENGTH_Y = float(P["length_x"]), float(P["length_y"])
LOAD_AMPLITUDE = float(P["load_amplitude"])
POISSON_RATIO = float(P["poisson_ratio"])
RIGIDITY = float(P["youngs_modulus"]) * float(P["thickness"])**3 / (
    12.0 * (1.0 - POISSON_RATIO**2)
)
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(CONFIG["sampling"]["boundary_points_per_edge"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
EPOCHS = int(CONFIG["training"]["epochs"])
LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu": return torch.device("cpu")
    if name == "gpu" and torch.cuda.is_available(): return torch.device("cuda")
    if name == "gpu": raise RuntimeError("GPU requested, but CUDA is not available")
    raise ValueError("device must be 'cpu' or 'gpu'")


def transverse_load(points):
    return LOAD_AMPLITUDE * torch.sin(math.pi*points[:, 0]/LENGTH_X) * torch.sin(
        math.pi*points[:, 1]/LENGTH_Y
    )


def exact_solution(points):
    k2 = (math.pi/LENGTH_X)**2 + (math.pi/LENGTH_Y)**2
    amplitude = LOAD_AMPLITUDE / (RIGIDITY*k2**2)
    return amplitude * torch.sin(math.pi*points[:, 0]/LENGTH_X) * torch.sin(
        math.pi*points[:, 1]/LENGTH_Y
    )


def sample_interior(count, device, generator=None):
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] *= LENGTH_X; points[:, 1] *= LENGTH_Y
    return points


def sample_boundary(count, device):
    x = torch.linspace(0.0, LENGTH_X, count, device=device)
    y = torch.linspace(0.0, LENGTH_Y, count, device=device)
    return {
        "left": torch.stack((torch.zeros_like(y), y), 1),
        "right": torch.stack((torch.full_like(y, LENGTH_X), y), 1),
        "bottom": torch.stack((x, torch.zeros_like(x)), 1),
        "top": torch.stack((x, torch.full_like(x, LENGTH_Y)), 1),
    }


def derivative(output, coordinates, axis, order):
    result = output
    for _ in range(order):
        result = torch.autograd.grad(
            result, coordinates, torch.ones_like(result), create_graph=True
        )[0][:, axis]
    return result


def plate_residual(model, points):
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    w_xxxx = derivative(solution, coordinates, 0, 4)
    w_yyyy = derivative(solution, coordinates, 1, 4)
    w_xx = derivative(solution, coordinates, 0, 2)
    w_xxyy = derivative(w_xx, coordinates, 1, 2)
    return (RIGIDITY*(w_xxxx + 2*w_xxyy + w_yyyy)-transverse_load(coordinates))/LOAD_AMPLITUDE


def boundary_terms(model, boundaries):
    def edge(points, vertical):
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(coordinates)
        w_xx = derivative(solution, coordinates, 0, 2)
        w_yy = derivative(solution, coordinates, 1, 2)
        curvature = w_xx + POISSON_RATIO*w_yy if vertical else w_yy + POISSON_RATIO*w_xx
        return solution, curvature
    data = [
        edge(boundaries["left"], True), edge(boundaries["right"], True),
        edge(boundaries["bottom"], False), edge(boundaries["top"], False),
    ]
    deflection = torch.stack([v.square().mean() for v, _ in data]).mean()
    center = torch.tensor([[LENGTH_X/2, LENGTH_Y/2]], device=next(model.parameters()).device)
    scale = LENGTH_X*LENGTH_Y/exact_solution(center)[0]
    moment = torch.stack([(scale*c).square().mean() for _, c in data]).mean()
    return deflection, moment


def pinn_loss(model, interior, boundaries):
    pde = plate_residual(model, interior).square().mean()
    if model.constraint_mode == "hard": return float(CONFIG["loss_weights"]["pde"])*pde
    deflection, moment = boundary_terms(model, boundaries)
    weights = CONFIG["loss_weights"]
    return float(weights["pde"])*pde + float(weights["deflection"])*deflection + float(weights["moment"])*moment


def relative_l2_error(model, points):
    with torch.no_grad(): prediction, reference = model(points), exact_solution(points)
    return torch.linalg.vector_norm(prediction-reference)/torch.linalg.vector_norm(reference)


class KirchhoffPINN(torch.nn.Module):
    def __init__(self, n_layers, n_neurons, constraint_mode):
        super().__init__()
        if constraint_mode not in ("soft", "hard"): raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.constraint_mode = constraint_mode
        modules = []; width = 2
        for _ in range(n_layers):
            layer = torch.nn.Linear(width, n_neurons)
            torch.nn.init.xavier_normal_(layer.weight); torch.nn.init.zeros_(layer.bias)
            modules.extend((layer, torch.nn.Tanh())); width = n_neurons
        output = torch.nn.Linear(width, 1)
        torch.nn.init.xavier_normal_(output.weight); torch.nn.init.zeros_(output.bias)
        modules.append(output); self.network = torch.nn.Sequential(*modules)

    def forward(self, points):
        x, y = points[:, 0], points[:, 1]
        if self.constraint_mode == "hard":
            ax, ay = math.pi*x/LENGTH_X, math.pi*y/LENGTH_Y
            raw = self.network(torch.stack((torch.cos(ax), torch.cos(ay)), 1)).squeeze(-1)
            return torch.sin(ax)*torch.sin(ay)*raw
        normalized = torch.stack((2*x/LENGTH_X-1, 2*y/LENGTH_Y-1), 1)
        return self.network(normalized).squeeze(-1)


def load_hyperparameters(path=RESULT_PATH):
    params = json.loads(path.read_text(encoding="utf-8"))["best_params"]
    return int(params["n_layers"]), int(params["n_neurons"])


def save_results(model, history, validation, error, wall_time, metadata):
    resolution = int(CONFIG["sampling"]["plot_points_per_axis"])
    x, y = np.linspace(0, LENGTH_X, resolution), np.linspace(0, LENGTH_Y, resolution)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    device = next(model.parameters()).device
    points = torch.tensor(np.column_stack((gx.ravel(), gy.ravel())), dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(gx.shape)
        reference = exact_solution(points).cpu().numpy().reshape(gx.shape)
        vp, vr = model(validation).cpu().numpy(), exact_solution(validation).cpu().numpy()
    absolute_error = np.abs(prediction-reference)
    output = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]; output.mkdir(parents=True, exist_ok=True)
    stem = f"pinn_kirchhoff_{model.constraint_mode}"
    solution_path, training_path = output/f"{stem}_solution_data.npz", output/f"{stem}_training_data.npz"
    np.savez_compressed(solution_path, x=x, y=y, prediction=prediction, reference=reference,
        absolute_error=absolute_error, validation_relative_l2_error=error,
        plot_grid_relative_l2_error=np.linalg.norm(prediction-reference)/np.linalg.norm(reference),
        validation_points=validation.cpu().numpy(), validation_prediction=vp,
        validation_reference=vr, training_wall_time_seconds=wall_time,
        constraint_mode=np.asarray(model.constraint_mode), **metadata)
    losses = np.asarray(history)
    np.savez_compressed(training_path, epoch=np.arange(1, losses.size+1), loss=losses,
        training_wall_time_seconds=wall_time)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, field, title in zip(axes, (prediction, reference, absolute_error), ("PINN prediction", "Exact", "Absolute error")):
        image=axis.contourf(gx, gy, field, levels=40); fig.colorbar(image, ax=axis); axis.set_title(title); axis.set_aspect("equal")
    fig.savefig(output/f"{stem}_solution.png", dpi=300)
    tf, ta = plt.subplots(); ta.semilogy(np.arange(1,losses.size+1), losses); ta.set(xlabel="Epoch", ylabel="Loss"); tf.tight_layout(); tf.savefig(output/f"{stem}_training.png",dpi=300)
    if CONFIG["output"]["show_plots"]: plt.show()
    else: plt.close("all")


def main(device_name=DEFAULT_DEVICE, n_layers=N_LAYERS, n_neurons=N_NEURONS,
         constraint_mode=CONSTRAINT_MODE, epochs=EPOCHS, use_tuned=None,
         initialization_seed=SEED, sampling_seed=SEED, validation_seed=None,
         save_outputs=True, verbose=True):
    set_seed(initialization_seed); device=select_device(device_name)
    should_load = LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load: n_layers,n_neurons=load_hyperparameters()
    model=KirchhoffPINN(n_layers,n_neurons,constraint_mode).to(device)
    generator=torch.Generator(device=device).manual_seed(sampling_seed)
    interior=sample_interior(N_INTERIOR_POINTS,device,generator)
    boundaries=sample_boundary(N_BOUNDARY_POINTS_PER_EDGE,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    cfg=CONFIG["training"]; scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=cfg["scheduler"]["factor"],patience=cfg["scheduler"]["patience"],min_lr=cfg["scheduler"]["min_learning_rate"])
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    history=train(model.parameters(),lambda:pinn_loss(model,interior,boundaries),optimizer,epochs=epochs,scheduler=scheduler,early_stopping_patience=cfg["early_stopping"]["patience"],min_delta=cfg["early_stopping"]["min_delta"],print_every=cfg["print_every"] if verbose else 0)
    if device.type=="cuda": torch.cuda.synchronize(device)
    wall_time=time.perf_counter()-start
    vg=generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed)
    validation=sample_interior(N_VALIDATION_POINTS,device,vg); error=float(relative_l2_error(model,validation))
    if verbose:
        d,m=boundary_terms(model,boundaries); print(f"PINN: layers={n_layers}, neurons={n_neurons}, constraints={constraint_mode}, device={device}"); print(f"Training wall time: {wall_time:.3f} s"); print(f"Validation relative L2 error: {error:.6e}"); print(f"Boundary RMS deflection: {math.sqrt(float(d.detach())):.6e}"); print(f"Normalized boundary RMS moment: {math.sqrt(float(m.detach())):.6e}")
    if save_outputs: save_results(model,history,validation,error,wall_time,{"n_layers":n_layers,"n_neurons":n_neurons})
    return {"validation_relative_l2_error":error,"training_time_seconds":wall_time}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--device",choices=("cpu","gpu"),default=DEFAULT_DEVICE); parser.add_argument("--constraint",choices=("soft","hard"),default=CONSTRAINT_MODE); parser.add_argument("--epochs",type=int,default=EPOCHS)
    tuned=parser.add_mutually_exclusive_group(); tuned.add_argument("--use-tuned",dest="use_tuned",action="store_true"); tuned.add_argument("--no-use-tuned",dest="use_tuned",action="store_false"); parser.set_defaults(use_tuned=None); args=parser.parse_args(); main(args.device,constraint_mode=args.constraint,epochs=args.epochs,use_tuned=args.use_tuned)
