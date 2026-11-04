"""Physics-informed KAN for the manufactured (2+1)-D Klein--Gordon PDE."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from kan import KAN
except ImportError:
    KAN = None

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
    return (x+y)*torch.cos(2.0*time)+x*y*torch.sin(2.0*time)


def initial_value_xy(x,y): return x+y
def initial_velocity_xy(x,y): return 2.0*x*y


def forcing(points):
    value=exact_solution(points); return value.square()-4.0*value


def coons_lifting(points,function):
    x,y,time=points.unbind(1); sx=(x+1.0)/2.0; sy=(y+1.0)/2.0
    blend=lambda z:z.square()/(z.square()+(1.0-z).square()); hx,hy=blend(sx),blend(sy)
    make=lambda xx,yy:function(torch.stack((xx,yy,time),1))
    left,right=make(torch.full_like(x,-1.0),y),make(torch.full_like(x,1.0),y)
    bottom,top=make(x,torch.full_like(y,-1.0)),make(x,torch.full_like(y,1.0))
    c00=make(torch.full_like(x,-1.0),torch.full_like(y,-1.0)); c10=make(torch.full_like(x,1.0),torch.full_like(y,-1.0))
    c01=make(torch.full_like(x,-1.0),torch.full_like(y,1.0)); c11=make(torch.full_like(x,1.0),torch.full_like(y,1.0))
    return ((1-hx)*left+hx*right+(1-hy)*bottom+hy*top-((1-hx)*(1-hy)*c00+hx*(1-hy)*c10+(1-hx)*hy*c01+hx*hy*c11))


def hard_constraint_transform(points,raw,enforce_initial_velocity=False):
    x,y,time=points.unbind(1); boundary=coons_lifting(points,exact_solution)
    initial_points=torch.stack((x,y,torch.zeros_like(time)),1)
    initial_correction=initial_value_xy(x,y)-coons_lifting(initial_points,exact_solution)
    sx,sy,tau=(x+1.0)/2.0,(y+1.0)/2.0,time/10.0; distance=16.0*sx*(1-sx)*sy*(1-sy)
    if enforce_initial_velocity:
        velocity_function=lambda p:initial_velocity_xy(p[:,0],p[:,1])
        velocity_correction=initial_velocity_xy(x,y)-coons_lifting(initial_points,velocity_function)
        return boundary+initial_correction+time*velocity_correction+tau.square()*distance*raw
    return boundary+initial_correction+tau*distance*raw


def sample_interior(count,device,generator=None):
    points=torch.rand(count,3,device=device,generator=generator); points[:,:2]=-1.0+2.0*points[:,:2]; points[:,2]=10.0*points[:,2]; return points


def sample_constraints(face_count,initial_count,device,generator=None):
    xy=-1.0+2.0*torch.rand(initial_count,2,device=device,generator=generator)
    initial=torch.column_stack((xy,torch.zeros(initial_count,device=device))); faces=[]
    for axis,value in ((0,-1.0),(0,1.0),(1,-1.0),(1,1.0)):
        face=sample_interior(face_count,device,generator); face[:,axis]=value; faces.append(face)
    return {"initial":initial,"boundary":torch.cat(faces)}


def pde_residual(model,points):
    coordinates=points.detach().clone().requires_grad_(True); value=model(coordinates)
    first=torch.autograd.grad(value,coordinates,torch.ones_like(value),create_graph=True)[0]
    second=[torch.autograd.grad(first[:,axis],coordinates,torch.ones_like(first[:,axis]),create_graph=True)[0][:,axis] for axis in range(3)]
    return second[2]-second[0]-second[1]+value.square()-forcing(coordinates)


def physics_loss(model,interior,constraints,weights):
    pde=pde_residual(model,interior).square().mean()
    if model.constraint_mode=="hard": return weights["pde"]*pde
    initial=constraints["initial"].detach().clone().requires_grad_(True); prediction=model(initial)
    initial_value=(prediction-initial_value_xy(initial[:,0],initial[:,1])).square().mean()
    if model.enforce_initial_velocity:
        velocity=torch.autograd.grad(prediction,initial,torch.ones_like(prediction),create_graph=True)[0][:,2]
        initial_velocity=(velocity-initial_velocity_xy(initial[:,0],initial[:,1])).square().mean()
    else: initial_velocity=torch.zeros((),device=interior.device)
    boundary=(model(constraints["boundary"])-exact_solution(constraints["boundary"])).square().mean()
    return weights["pde"]*pde+weights["initial_value"]*initial_value+weights["initial_velocity"]*initial_velocity+weights["boundary"]*boundary


def relative_l2_error(model,points):
    with torch.no_grad(): reference,prediction=exact_solution(points),model(points)
    return torch.linalg.vector_norm(prediction-reference)/torch.linalg.vector_norm(reference)


def constraint_errors(model,constraints):
    initial=constraints["initial"].detach().clone().requires_grad_(True); prediction=model(initial)
    velocity=torch.autograd.grad(prediction,initial,torch.ones_like(prediction))[0][:,2]
    value_error=(prediction-exact_solution(initial)).abs().max().detach(); velocity_error=(velocity-initial_velocity_xy(initial[:,0],initial[:,1])).abs().max().detach()
    with torch.no_grad(): boundary_error=(model(constraints["boundary"])-exact_solution(constraints["boundary"])).abs().max()
    return value_error,velocity_error,boundary_error


def save_results(model,history,relative_error,validation,wall_time,output,stem,title,spatial_points,plot_times,metadata,show_plots):
    device=next(model.parameters()).device; x=np.linspace(*X_BOUNDS,spatial_points); y=np.linspace(*Y_BOUNDS,spatial_points)
    times=np.asarray(plot_times); gx,gy=np.meshgrid(x,y,indexing="xy"); prediction=[]; reference=[]
    with torch.no_grad():
        for value in times:
            points=torch.tensor(np.column_stack((gx.ravel(),gy.ravel(),np.full(gx.size,value))),dtype=torch.float32,device=device)
            prediction.append(model(points).cpu().numpy().reshape(gx.shape)); reference.append(exact_solution(points).cpu().numpy().reshape(gx.shape))
    prediction,reference=np.stack(prediction),np.stack(reference); box_time=np.linspace(*T_BOUNDS,spatial_points)
    vt,vy,vx=np.meshgrid(box_time,y,x,indexing="ij"); volume_points=torch.tensor(np.column_stack((vx.ravel(),vy.ravel(),vt.ravel())),dtype=torch.float32,device=device)
    predicted_chunks=[]; reference_chunks=[]
    with torch.no_grad():
        for start in range(0,volume_points.shape[0],32768):
            chunk=volume_points[start:start+32768]; predicted_chunks.append(model(chunk).cpu()); reference_chunks.append(exact_solution(chunk).cpu())
        validation_prediction=model(validation).cpu().numpy(); validation_reference=exact_solution(validation).cpu().numpy()
    volume_prediction=torch.cat(predicted_chunks).numpy().reshape(vt.shape); volume_reference=torch.cat(reference_chunks).numpy().reshape(vt.shape)
    output.mkdir(parents=True,exist_ok=True); payload=dict(x=x,y=y,time=times,prediction=prediction,reference=reference,absolute_error=np.abs(prediction-reference),box_time=box_time,volume_prediction=volume_prediction,volume_reference=volume_reference,volume_error=np.abs(volume_prediction-volume_reference),validation_points=validation.cpu().numpy(),validation_prediction=validation_prediction,validation_reference=validation_reference,validation_relative_l2_error=relative_error,training_wall_time_seconds=wall_time)
    payload.update({key:np.asarray(value) for key,value in metadata.items()}); np.savez_compressed(output/f"{stem}_solution_data.npz",**payload)
    loss=np.asarray(history); np.savez_compressed(output/f"{stem}_training_data.npz",epoch=np.arange(1,loss.size+1),loss=loss,training_wall_time_seconds=wall_time)
    print(f"Saved solution data to: {output/f'{stem}_solution_data.npz'}"); print(f"Saved training data to: {output/f'{stem}_training_data.npz'}")

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "pikan_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
GRID = int(CONFIG["model"]["grid"])
SPLINE_ORDER = int(CONFIG["model"]["spline_order"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
SAMPLING = CONFIG["sampling"]
TRAINING = CONFIG["training"]
LOSS_WEIGHTS = {key: float(value) for key, value in CONFIG["loss_weights"].items()}
ENFORCE_INITIAL_VELOCITY = bool(CONFIG["problem"].get("enforce_initial_velocity", False))
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]


class KleinGordonPIKAN(torch.nn.Module):
    def __init__(
        self, n_layers: int, n_neurons: int, grid: int, spline_order: int,
        constraint_mode: str, device: torch.device, seed: int = SEED,
        enforce_initial_velocity: bool = ENFORCE_INITIAL_VELOCITY,
    ) -> None:
        super().__init__()
        if KAN is None:
            raise ImportError("PIKAN requires pykan (import name 'kan'). Install pykan first.")
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_layers, self.n_neurons = n_layers, n_neurons
        self.grid, self.spline_order = grid, spline_order
        self.constraint_mode = constraint_mode
        self.enforce_initial_velocity = enforce_initial_velocity
        self.network = KAN(
            width=[3] + [n_neurons] * n_layers + [1], grid=grid,
            k=spline_order, seed=seed, device=device, auto_save=False,
            save_act=False, symbolic_enabled=False,
        )
        self.register_buffer("lower_bound", torch.tensor((X_BOUNDS[0], Y_BOUNDS[0], T_BOUNDS[0])))
        self.register_buffer("upper_bound", torch.tensor((X_BOUNDS[1], Y_BOUNDS[1], T_BOUNDS[1])))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        return hard_constraint_transform(points, raw, self.enforce_initial_velocity)


def pikan_loss(model, interior, constraints, weights=LOSS_WEIGHTS):
    return physics_loss(model, interior, constraints, weights)


def load_hyperparameters(mode: str, path: Path = RESULT_PATH) -> tuple[int, int, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mode in payload:
        payload = payload[mode]
    params = payload.get("best_params", payload)
    if not params:
        raise ValueError(f"No tuned {mode} PIKAN parameters in {path}")
    return int(params["n_layers"]), int(params["n_neurons"]), int(params["grid"]), int(params["spline_order"])


def main(
    device_name: str = DEFAULT_DEVICE, n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS, grid: int = GRID,
    spline_order: int = SPLINE_ORDER, constraint_mode: str = CONSTRAINT_MODE,
    epochs: int = int(TRAINING["epochs"]), use_tuned: bool | None = None,
    initialization_seed: int = SEED, sampling_seed: int = SEED,
    validation_seed: int | None = None, save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    set_seed(initialization_seed)
    device = select_device(device_name)
    should_load = LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists() and RESULT_PATH.stat().st_size > 3 if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons, grid, spline_order = load_hyperparameters(constraint_mode)
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Klein-Gordon PIKAN: layers={n_layers}, neurons={n_neurons}, grid={grid}, order={spline_order}, constraints={constraint_mode}")
    model = KleinGordonPIKAN(n_layers, n_neurons, grid, spline_order, constraint_mode, device, initialization_seed).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(int(SAMPLING["interior_points"]), device, generator)
    constraints = sample_constraints(int(SAMPLING["boundary_points_per_face"]), int(SAMPLING["initial_points"]), device, generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(TRAINING["learning_rate"]))
    scheduler_cfg = TRAINING["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]), min_lr=float(scheduler_cfg["min_learning_rate"]),
    )
    stopping = TRAINING["early_stopping"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train(
        model.parameters(), lambda: pikan_loss(model, interior, constraints), optimizer,
        epochs=epochs, scheduler=scheduler, early_stopping_patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]), print_every=int(TRAINING["print_every"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(int(SAMPLING["validation_points"]), device, validation_generator)
    relative_error = relative_l2_error(model, validation)
    residual_mse = pde_residual(model, validation).detach().square().mean()
    value_error, velocity_error, boundary_error = constraint_errors(model, constraints)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Training wall time: {wall_time:.3f} s")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-value error: {value_error.item():.6e}")
    velocity_label = "Maximum initial-velocity error" if ENFORCE_INITIAL_VELOCITY else "Initial-velocity diagnostic (not enforced)"
    print(f"{velocity_label}: {velocity_error.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    result = {"validation_relative_l2_error": relative_error.item(), "training_time_seconds": wall_time}
    if not save_outputs:
        return result
    save_results(
        model, history, relative_error.item(), validation, wall_time, POSTPROCESSING_DIR,
        f"pikan_klein_gordon_{constraint_mode}", f"Klein-Gordon PIKAN ({constraint_mode})",
        int(SAMPLING["plot_spatial_points"]), list(SAMPLING["plot_times"]),
        {"constraint_mode": constraint_mode, "n_layers": n_layers, "n_neurons": n_neurons, "grid": grid, "spline_order": spline_order},
        bool(CONFIG["output"]["show_plots"]),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the (2+1)-D Klein-Gordon equation with a PIKAN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--spline-order", type=int, default=SPLINE_ORDER)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.layers, args.neurons, args.grid, args.spline_order, args.constraint, args.epochs, args.use_tuned)
