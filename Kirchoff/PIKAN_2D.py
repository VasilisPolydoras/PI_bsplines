"""Physics-informed KAN for a simply supported Kirchhoff plate."""

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
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from training import train
try:
    from kan import KAN
except ImportError:
    KAN = None

SCRIPT_DIR=Path(__file__).resolve().parent
CONFIG=json.loads((SCRIPT_DIR/"pikan_config.json").read_text(encoding="utf-8"))
SEED=int(CONFIG["reproducibility"]["seed"]); DEFAULT_DEVICE=str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE=bool(CONFIG["runtime"]["load_tuned_if_available"])
P=CONFIG["problem"]; LENGTH_X=float(P["length_x"]); LENGTH_Y=float(P["length_y"])
LOAD_AMPLITUDE=float(P["load_amplitude"]); POISSON_RATIO=float(P["poisson_ratio"])
RIGIDITY=float(P["youngs_modulus"])*float(P["thickness"])**3/(12*(1-POISSON_RATIO**2))
N_LAYERS=int(CONFIG["model"]["n_layers"]); N_NEURONS=int(CONFIG["model"]["n_neurons"])
GRID=int(CONFIG["model"]["grid"]); SPLINE_ORDER=int(CONFIG["model"]["spline_order"])
CONSTRAINT_MODE=str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS=int(CONFIG["sampling"]["interior_points"]); N_BOUNDARY_POINTS_PER_EDGE=int(CONFIG["sampling"]["boundary_points_per_edge"]); N_VALIDATION_POINTS=int(CONFIG["sampling"]["validation_points"])
EPOCHS=int(CONFIG["training"]["epochs"]); LEARNING_RATE=float(CONFIG["training"]["learning_rate"])
RESULT_PATH=SCRIPT_DIR/CONFIG["output"]["tuned_hyperparameters"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name=="cpu": return torch.device("cpu")
    if name=="gpu" and torch.cuda.is_available(): return torch.device("cuda")
    if name=="gpu": raise RuntimeError("GPU requested, but CUDA is not available")
    raise ValueError("device must be 'cpu' or 'gpu'")


def transverse_load(points):
    return LOAD_AMPLITUDE*torch.sin(math.pi*points[:,0]/LENGTH_X)*torch.sin(math.pi*points[:,1]/LENGTH_Y)


def exact_solution(points):
    k2=(math.pi/LENGTH_X)**2+(math.pi/LENGTH_Y)**2
    return LOAD_AMPLITUDE/(RIGIDITY*k2**2)*torch.sin(math.pi*points[:,0]/LENGTH_X)*torch.sin(math.pi*points[:,1]/LENGTH_Y)


def sample_interior(count,device,generator=None):
    points=torch.rand(count,2,device=device,generator=generator); points[:,0]*=LENGTH_X; points[:,1]*=LENGTH_Y; return points


def sample_boundary(count,device):
    x=torch.linspace(0,LENGTH_X,count,device=device); y=torch.linspace(0,LENGTH_Y,count,device=device)
    return {"left":torch.stack((torch.zeros_like(y),y),1),"right":torch.stack((torch.full_like(y,LENGTH_X),y),1),"bottom":torch.stack((x,torch.zeros_like(x)),1),"top":torch.stack((x,torch.full_like(x,LENGTH_Y)),1)}


def derivative(output,coordinates,axis,order):
    result=output
    for _ in range(order): result=torch.autograd.grad(result,coordinates,torch.ones_like(result),create_graph=True)[0][:,axis]
    return result


def plate_residual(model,points):
    c=points.detach().clone().requires_grad_(True); w=model(c)
    xxxx=derivative(w,c,0,4); yyyy=derivative(w,c,1,4); xx=derivative(w,c,0,2); xxyy=derivative(xx,c,1,2)
    return (RIGIDITY*(xxxx+2*xxyy+yyyy)-transverse_load(c))/LOAD_AMPLITUDE


def boundary_terms(model,boundaries):
    def edge(points,vertical):
        c=points.detach().clone().requires_grad_(True); w=model(c); xx=derivative(w,c,0,2); yy=derivative(w,c,1,2)
        return w, xx+POISSON_RATIO*yy if vertical else yy+POISSON_RATIO*xx
    data=[edge(boundaries["left"],True),edge(boundaries["right"],True),edge(boundaries["bottom"],False),edge(boundaries["top"],False)]
    deflection=torch.stack([w.square().mean() for w,_ in data]).mean()
    center=torch.tensor([[LENGTH_X/2,LENGTH_Y/2]],device=next(model.parameters()).device); scale=LENGTH_X*LENGTH_Y/exact_solution(center)[0]
    moment=torch.stack([(scale*c).square().mean() for _,c in data]).mean(); return deflection,moment


def pikan_loss(model,interior,boundaries):
    pde=plate_residual(model,interior).square().mean(); weights=CONFIG["loss_weights"]
    if model.constraint_mode=="hard": return float(weights["pde"])*pde
    d,m=boundary_terms(model,boundaries); return float(weights["pde"])*pde+float(weights["deflection"])*d+float(weights["moment"])*m


def relative_l2_error(model,points):
    with torch.no_grad(): p=model(points); r=exact_solution(points)
    return torch.linalg.vector_norm(p-r)/torch.linalg.vector_norm(r)


class KirchhoffPIKAN(torch.nn.Module):
    def __init__(self,n_layers,n_neurons,grid,spline_order,constraint_mode,device,seed):
        super().__init__()
        if KAN is None: raise ImportError("PIKAN requires pykan (import name 'kan')")
        if constraint_mode not in ("soft","hard"): raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.constraint_mode=constraint_mode
        self.network=KAN(width=[2]+[n_neurons]*n_layers+[1],grid=grid,k=spline_order,seed=seed,device=device,auto_save=False,save_act=False,symbolic_enabled=False)

    def forward(self,points):
        x,y=points[:,0],points[:,1]
        if self.constraint_mode=="hard":
            ax,ay=math.pi*x/LENGTH_X,math.pi*y/LENGTH_Y
            raw=self.network(torch.stack((torch.cos(ax),torch.cos(ay)),1)).squeeze(-1)
            return torch.sin(ax)*torch.sin(ay)*raw
        return self.network(torch.stack((2*x/LENGTH_X-1,2*y/LENGTH_Y-1),1)).squeeze(-1)


def load_hyperparameters(path=RESULT_PATH):
    p=json.loads(path.read_text(encoding="utf-8"))["best_params"]
    return tuple(int(p[k]) for k in ("n_layers","n_neurons","grid","spline_order"))


def save_results(model,history,validation,error,wall_time,metadata):
    resolution=int(CONFIG["sampling"]["plot_points_per_axis"]); x=np.linspace(0,LENGTH_X,resolution); y=np.linspace(0,LENGTH_Y,resolution); gx,gy=np.meshgrid(x,y,indexing="xy")
    device=next(model.parameters()).device; points=torch.tensor(np.column_stack((gx.ravel(),gy.ravel())),dtype=torch.float32,device=device)
    with torch.no_grad():
        prediction=model(points).cpu().numpy().reshape(gx.shape); reference=exact_solution(points).cpu().numpy().reshape(gx.shape); vp=model(validation).cpu().numpy(); vr=exact_solution(validation).cpu().numpy()
    error_map=np.abs(prediction-reference); output=SCRIPT_DIR/CONFIG["output"]["postprocessing_directory"]; output.mkdir(parents=True,exist_ok=True); stem=f"pikan_kirchhoff_{model.constraint_mode}"
    solution_path=output/f"{stem}_solution_data.npz"; training_path=output/f"{stem}_training_data.npz"
    np.savez_compressed(solution_path,x=x,y=y,prediction=prediction,reference=reference,absolute_error=error_map,validation_relative_l2_error=error,plot_grid_relative_l2_error=np.linalg.norm(prediction-reference)/np.linalg.norm(reference),validation_points=validation.cpu().numpy(),validation_prediction=vp,validation_reference=vr,training_wall_time_seconds=wall_time,constraint_mode=np.asarray(model.constraint_mode),**metadata)
    losses=np.asarray(history); np.savez_compressed(training_path,epoch=np.arange(1,losses.size+1),loss=losses,training_wall_time_seconds=wall_time)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for axis,field,title in zip(axes,(prediction,reference,error_map),("PIKAN prediction","Exact","Absolute error")):
        image=axis.contourf(gx,gy,field,levels=40); fig.colorbar(image,ax=axis); axis.set_title(title); axis.set_aspect("equal")
    fig.savefig(output/f"{stem}_solution.png",dpi=300); tf,ta=plt.subplots(); ta.semilogy(np.arange(1,losses.size+1),losses); ta.set(xlabel="Epoch",ylabel="Loss"); tf.tight_layout(); tf.savefig(output/f"{stem}_training.png",dpi=300)
    if CONFIG["output"]["show_plots"]: plt.show()
    else: plt.close("all")


def main(device_name=DEFAULT_DEVICE,n_layers=N_LAYERS,n_neurons=N_NEURONS,grid=GRID,spline_order=SPLINE_ORDER,constraint_mode=CONSTRAINT_MODE,epochs=EPOCHS,use_tuned=None,initialization_seed=SEED,sampling_seed=SEED,validation_seed=None,save_outputs=True,verbose=True):
    set_seed(initialization_seed); device=select_device(device_name); should_load=LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load: n_layers,n_neurons,grid,spline_order=load_hyperparameters()
    model=KirchhoffPIKAN(n_layers,n_neurons,grid,spline_order,constraint_mode,device,initialization_seed).to(device); generator=torch.Generator(device=device).manual_seed(sampling_seed); interior=sample_interior(N_INTERIOR_POINTS,device,generator); boundaries=sample_boundary(N_BOUNDARY_POINTS_PER_EDGE,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); cfg=CONFIG["training"]; scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=cfg["scheduler"]["factor"],patience=cfg["scheduler"]["patience"],min_lr=cfg["scheduler"]["min_learning_rate"])
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter(); history=train(model.parameters(),lambda:pikan_loss(model,interior,boundaries),optimizer,epochs=epochs,scheduler=scheduler,early_stopping_patience=cfg["early_stopping"]["patience"],min_delta=cfg["early_stopping"]["min_delta"],print_every=cfg["print_every"] if verbose else 0)
    if device.type=="cuda": torch.cuda.synchronize(device)
    wall_time=time.perf_counter()-start; vg=generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed); validation=sample_interior(N_VALIDATION_POINTS,device,vg); error=float(relative_l2_error(model,validation))
    if verbose:
        d,m=boundary_terms(model,boundaries); print(f"PIKAN: layers={n_layers}, neurons={n_neurons}, grid={grid}, order={spline_order}, constraints={constraint_mode}, device={device}"); print(f"Training wall time: {wall_time:.3f} s"); print(f"Validation relative L2 error: {error:.6e}"); print(f"Boundary RMS deflection: {math.sqrt(float(d.detach())):.6e}"); print(f"Normalized boundary RMS moment: {math.sqrt(float(m.detach())):.6e}")
    if save_outputs: save_results(model,history,validation,error,wall_time,{"n_layers":n_layers,"n_neurons":n_neurons,"grid":grid,"spline_order":spline_order})
    return {"validation_relative_l2_error":error,"training_time_seconds":wall_time}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--device",choices=("cpu","gpu"),default=DEFAULT_DEVICE); parser.add_argument("--constraint",choices=("soft","hard"),default=CONSTRAINT_MODE); parser.add_argument("--epochs",type=int,default=EPOCHS); tuned=parser.add_mutually_exclusive_group(); tuned.add_argument("--use-tuned",dest="use_tuned",action="store_true"); tuned.add_argument("--no-use-tuned",dest="use_tuned",action="store_false"); parser.set_defaults(use_tuned=None); a=parser.parse_args(); main(a.device,constraint_mode=a.constraint,epochs=a.epochs,use_tuned=a.use_tuned)
