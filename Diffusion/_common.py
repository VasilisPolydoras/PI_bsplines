"""Shared utilities for the forward one-dimensional diffusion benchmark."""
from __future__ import annotations
import json,math,random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
def load_config(name):return json.loads((ROOT/name).read_text(encoding="utf-8"))
def set_seed(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def device_from(name):
    if name=="cpu":return torch.device("cpu")
    if name=="gpu" and torch.cuda.is_available():return torch.device("cuda")
    if name=="gpu":raise RuntimeError("GPU requested, but CUDA is unavailable")
    raise ValueError("device must be 'cpu' or 'gpu'")
def exact(points,alpha):return torch.sin(2*math.pi*points[:,0])*torch.exp(-4*math.pi**2*alpha*points[:,1])
def sample(count,device,generator=None):return torch.rand(count,2,device=device,generator=generator)
def boundary(count,device,generator=None):
    t=torch.rand(count,device=device,generator=generator);return torch.cat((torch.stack((torch.zeros_like(t),t),1),torch.stack((torch.ones_like(t),t),1)))
def initial(count,device):
    x=torch.linspace(0,1,count,device=device);return torch.stack((x,torch.zeros_like(x)),1)
def derivative(y,x,axis,order=1):
    out=y
    for _ in range(order):out=torch.autograd.grad(out,x,torch.ones_like(out),create_graph=True)[0][:,axis]
    return out
def residual(model,points,alpha):
    z=points.detach().clone().requires_grad_(True);u=model(z);return derivative(u,z,1)-alpha*derivative(u,z,0,2)
def physics_loss(model,interior,boundary_points,initial_points,alpha,weights):
    pde=residual(model,interior,alpha).square().mean()
    if model.constraint_mode=="hard":return float(weights["pde"])*pde
    bc=model(boundary_points).square().mean();ic=(model(initial_points)-exact(initial_points,alpha)).square().mean()
    return float(weights["pde"])*pde+float(weights["boundary"])*bc+float(weights["initial"])*ic
def relative_l2(model,points,alpha):
    with torch.no_grad():pred=model(points);ref=exact(points,alpha)
    return torch.linalg.vector_norm(pred-ref)/torch.linalg.vector_norm(ref)
def train(model,fn,cfg):
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["learning_rate"]));sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=float(cfg["scheduler"]["factor"]),patience=int(cfg["scheduler"]["patience"]),min_lr=float(cfg["scheduler"]["min_learning_rate"]));best=float("inf");stale=0;state=None;history=[]
    for epoch in range(int(cfg["epochs"])):
        opt.zero_grad();value=fn();number=float(value.detach());value.backward();opt.step();sch.step(number);history.append(number)
        if number<best-float(cfg["early_stopping"]["min_delta"]):best=number;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if int(cfg["print_every"]) and (epoch+1)%int(cfg["print_every"])==0:print(f"Epoch [{epoch+1}/{cfg['epochs']}], loss={number:.6e}")
        if stale>=int(cfg["early_stopping"]["patience"]):break
    if state:model.load_state_dict(state)
    return history
def save(model,history,error,config,alpha,method):
    post=ROOT/config["output"]["postprocessing_directory"];post.mkdir(parents=True,exist_ok=True);n=int(config["sampling"]["plot_points_per_axis"]);x=np.linspace(0,1,n);t=np.linspace(0,1,n);gx,gt=np.meshgrid(x,t,indexing="xy");device=next(model.parameters()).device;points=torch.tensor(np.column_stack((gx.ravel(),gt.ravel())),dtype=torch.float32,device=device)
    with torch.no_grad():pred=model(points).cpu().numpy().reshape(gx.shape);ref=exact(points,alpha).cpu().numpy().reshape(gx.shape)
    stem=f"{method}_diffusion_{model.constraint_mode}";np.savez_compressed(post/f"{stem}_solution_data.npz",x=x,t=t,prediction=pred,reference=ref,absolute_error=np.abs(pred-ref),validation_relative_l2_error=error);np.savez_compressed(post/f"{stem}_training_data.npz",epoch=np.arange(1,len(history)+1),loss=history);plot_solution(post/f"{stem}_solution_data.npz",post/f"{stem}_solution.png",method.upper());plot_training(post/f"{stem}_training_data.npz",post/f"{stem}_training.png",method.upper());plt.show() if config["output"]["show_plots"] else plt.close("all")
def plot_solution(data_path,out,title):
    with np.load(data_path) as d:x,t=d["x"],d["t"];fields=(d["prediction"],d["reference"],d["absolute_error"])
    gx,gt=np.meshgrid(x,t,indexing="xy");fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for ax,z,label in zip(axes,fields,("Prediction","Exact","Absolute error")):im=ax.contourf(gx,gt,z,40,cmap="viridis");fig.colorbar(im,ax=ax);ax.set_title(label);ax.set_xlabel("x");ax.set_ylabel("t")
    fig.suptitle(title);fig.savefig(out,dpi=300,bbox_inches="tight")
def plot_training(data_path,out,title):
    with np.load(data_path) as d:e,l=d["epoch"],d["loss"]
    fig,ax=plt.subplots();ax.semilogy(e,l);ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.set_title(title);fig.tight_layout();fig.savefig(out,dpi=300,bbox_inches="tight")
