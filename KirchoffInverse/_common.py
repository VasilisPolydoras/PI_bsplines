"""Independent utilities for inverse identification of plate rigidity D."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parent


def initialize_positive_parameter(value: float, minimum: float, device) -> torch.nn.Parameter:
    """Return raw softplus parameter representing ``value > minimum``."""
    shifted = max(float(value) - float(minimum), torch.finfo(torch.float32).eps)
    raw = math.log(math.expm1(shifted))
    return torch.nn.Parameter(torch.tensor(raw, dtype=torch.float32, device=device))


def positive_parameter(raw: torch.Tensor, minimum: float) -> torch.Tensor:
    return float(minimum) + functional.softplus(raw)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from(name: str) -> torch.device:
    if name == "cpu": return torch.device("cpu")
    if name == "gpu" and torch.cuda.is_available(): return torch.device("cuda")
    if name == "gpu": raise RuntimeError("GPU requested, but CUDA is unavailable")
    raise ValueError("device must be 'cpu' or 'gpu'")


def problem(config: dict) -> dict[str, float]:
    p = config["problem"]
    true_d = float(p["youngs_modulus"]) * float(p["thickness"])**3 / (
        12.0 * (1.0 - float(p["poisson_ratio"])**2)
    )
    return {
        "length_x": float(p["length_x"]), "length_y": float(p["length_y"]),
        "load": float(p["load_amplitude"]), "nu": float(p["poisson_ratio"]),
        "true_d": true_d,
    }


def load_value(points: torch.Tensor, p: dict[str, float]) -> torch.Tensor:
    return p["load"] * torch.sin(math.pi * points[:, 0] / p["length_x"]) * torch.sin(
        math.pi * points[:, 1] / p["length_y"]
    )


def exact_solution(points: torch.Tensor, p: dict[str, float]) -> torch.Tensor:
    k2 = (math.pi / p["length_x"])**2 + (math.pi / p["length_y"])**2
    amplitude = p["load"] / (p["true_d"] * k2**2)
    return amplitude * torch.sin(math.pi * points[:, 0] / p["length_x"]) * torch.sin(
        math.pi * points[:, 1] / p["length_y"]
    )


def sample_points(count, p, device, generator=None):
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] *= p["length_x"]; points[:, 1] *= p["length_y"]
    return points


def sample_boundary(count, p, device):
    x = torch.linspace(0, p["length_x"], count, device=device)
    y = torch.linspace(0, p["length_y"], count, device=device)
    return {
        "left": torch.stack((torch.zeros_like(y), y), 1),
        "right": torch.stack((torch.full_like(y, p["length_x"]), y), 1),
        "bottom": torch.stack((x, torch.zeros_like(x)), 1),
        "top": torch.stack((x, torch.full_like(x, p["length_y"])), 1),
    }


def observations(count, noise_std, p, device, generator=None):
    points = sample_points(count, p, device, generator)
    exact = exact_solution(points, p)
    noise = noise_std * torch.randn(exact.shape, device=device, generator=generator)
    return points, exact + noise


def derivative(output, coordinates, axis, order):
    result = output
    for _ in range(order):
        result = torch.autograd.grad(
            result, coordinates, torch.ones_like(result), create_graph=True
        )[0][:, axis]
    return result


def residual(model, points, p):
    coordinates = points.detach().clone().requires_grad_(True)
    w = model(coordinates)
    w_xxxx = derivative(w, coordinates, 0, 4)
    w_yyyy = derivative(w, coordinates, 1, 4)
    w_xxyy = derivative(derivative(w, coordinates, 0, 2), coordinates, 1, 2)
    return (
        model.flexural_rigidity() * (w_xxxx + 2*w_xxyy + w_yyyy)
        - load_value(coordinates, p)
    ) / p["load"]


def boundary_loss(model, boundaries, p):
    displacement, moment = [], []
    for name, points in boundaries.items():
        coordinates = points.detach().clone().requires_grad_(True)
        w = model(coordinates)
        w_xx = derivative(w, coordinates, 0, 2)
        w_yy = derivative(w, coordinates, 1, 2)
        curvature = w_xx + p["nu"]*w_yy if name in ("left", "right") else w_yy + p["nu"]*w_xx
        displacement.append(w.square().mean())
        moment.append(curvature.square().mean())
    scale = p["length_x"] * p["length_y"] / exact_solution(
        torch.tensor([[p["length_x"]/2, p["length_y"]/2]], device=next(model.parameters()).device), p
    )[0]
    return torch.stack(displacement).mean(), torch.stack([(scale**2)*term for term in moment]).mean()


def loss(model, interior, sensor_points, sensor_values, boundaries, p, weights):
    pde = residual(model, interior, p).square().mean()
    scale = exact_solution(
        torch.tensor([[p["length_x"]/2, p["length_y"]/2]], device=interior.device), p
    )[0]
    data = ((model(sensor_points)-sensor_values)/scale).square().mean()
    total = float(weights["pde"])*pde + float(weights["data"])*data
    if model.constraint_mode == "soft":
        value_bc, moment_bc = boundary_loss(model, boundaries, p)
        total = total + float(weights["deflection"])*value_bc + float(weights["moment"])*moment_bc
    return total


def relative_l2(model, points, p):
    with torch.no_grad():
        prediction=model(points); reference=exact_solution(points,p)
    return torch.linalg.vector_norm(prediction-reference)/torch.linalg.vector_norm(reference)


def train_inverse(model, loss_fn, config):
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"]))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,factor=float(config["scheduler"]["factor"]),
        patience=int(config["scheduler"]["patience"]),
        min_lr=float(config["scheduler"]["min_learning_rate"]),
    )
    best=float("inf"); stale=0; best_state=None; losses=[]; d_values=[]
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(); objective=loss_fn(); value=float(objective.detach())
        objective.backward(); optimizer.step(); scheduler.step(value)
        losses.append(value); d_values.append(float(model.flexural_rigidity().detach()))
        if value < best-float(config["early_stopping"]["min_delta"]):
            best=value; stale=0
            best_state={key: item.detach().clone() for key,item in model.state_dict().items()}
        else: stale+=1
        if int(config["print_every"]) and (epoch+1)%int(config["print_every"])==0:
            print(f"Epoch [{epoch+1}/{config['epochs']}], loss={value:.6e}, D={d_values[-1]:.6f}")
        if stale>=int(config["early_stopping"]["patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    return losses,d_values


def save_results(model, losses, d_history, validation_error, config, p, method, sensors):
    post=ROOT/config["output"]["postprocessing_directory"]; post.mkdir(parents=True,exist_ok=True)
    n=int(config["sampling"]["plot_points_per_axis"])
    x=np.linspace(0,p["length_x"],n); y=np.linspace(0,p["length_y"],n)
    gx,gy=np.meshgrid(x,y,indexing="xy"); device=next(model.parameters()).device
    points=torch.tensor(np.column_stack((gx.ravel(),gy.ravel())),dtype=torch.float32,device=device)
    with torch.no_grad():
        pred=model(points).cpu().numpy().reshape(gx.shape)
        ref=exact_solution(points,p).cpu().numpy().reshape(gx.shape)
    mode=model.constraint_mode; stem=f"{method}_kirchhoff_inverse_{mode}"
    estimated=float(model.flexural_rigidity().detach())
    np.savez_compressed(
        post/f"{stem}_solution_data.npz",x=x,y=y,prediction=pred,reference=ref,
        absolute_error=np.abs(pred-ref),sensor_points=sensors[0].detach().cpu().numpy(),
        sensor_values=sensors[1].detach().cpu().numpy(),estimated_d=estimated,
        true_d=p["true_d"],parameter_relative_error=abs(estimated-p["true_d"])/p["true_d"],
        validation_relative_l2_error=validation_error,
    )
    np.savez_compressed(post/f"{stem}_training_data.npz",epoch=np.arange(1,len(losses)+1),loss=losses,estimated_d=d_history)
    plot_solution(post/f"{stem}_solution_data.npz",post/f"{stem}_solution.png",f"{method.upper()} ({mode})")
    plot_training(post/f"{stem}_training_data.npz",post/f"{stem}_training.png",f"{method.upper()} ({mode})",p["true_d"])
    if config["output"]["show_plots"]: plt.show()
    else: plt.close("all")


def plot_solution(data_path,output_path,title):
    with np.load(data_path) as data:
        x,y=data["x"],data["y"]; fields=(data["prediction"],data["reference"],data["absolute_error"])
        estimated=float(data["estimated_d"]); true=float(data["true_d"])
    gx,gy=np.meshgrid(x,y,indexing="xy"); fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for ax,field,label in zip(axes,fields,("Prediction","Exact","Absolute error")):
        image=ax.contourf(gx,gy,field,40,cmap="viridis");fig.colorbar(image,ax=ax);ax.set_aspect("equal");ax.set_title(label);ax.set_xlabel("x");ax.set_ylabel("y")
    fig.suptitle(f"{title}: D={estimated:.6f} (true {true:.6f})");fig.savefig(output_path,dpi=300,bbox_inches="tight")


def plot_training(data_path,output_path,title,true_d):
    with np.load(data_path) as data: epoch,losses,d_values=data["epoch"],data["loss"],data["estimated_d"]
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].semilogy(epoch,losses);axes[0].set_title("Loss");axes[1].plot(epoch,d_values);axes[1].axhline(true_d,color="k",ls="--",label="true D");axes[1].legend();axes[1].set_title("Identified rigidity")
    for ax in axes: ax.set_xlabel("Epoch")
    fig.suptitle(title);fig.tight_layout();fig.savefig(output_path,dpi=300,bbox_inches="tight")
