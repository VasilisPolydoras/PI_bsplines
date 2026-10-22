"""PIKAN diffusion solver with hard or soft initial/boundary conditions."""
import argparse,json,torch
try:from kan import KAN
except ImportError:KAN=None
from _common import ROOT,boundary,device_from,initial,physics_loss,relative_l2,sample,save,set_seed,train
CONFIG=json.loads((ROOT/"pikan_config.json").read_text());RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]
class DiffusionPIKAN(torch.nn.Module):
    def __init__(self,layers,neurons,grid,k,mode,device,seed):
        super().__init__();
        if KAN is None:raise ImportError("PIKAN requires pykan (import name 'kan').")
        self.constraint_mode=mode;self.net=KAN(width=[2]+[neurons]*layers+[1],grid=grid,k=k,seed=seed,device=device,auto_save=False,save_act=False,symbolic_enabled=False)
    def forward(self,z):
        x,t=z[:,0],z[:,1];raw=self.net(torch.stack((2*x-1,2*t-1),1)).squeeze(-1);return torch.sin(2*torch.pi*x)+t*x*(1-x)*raw if self.constraint_mode=="hard" else raw
def architecture(mode):
    keys=("n_layers","n_neurons","grid","spline_order")
    if CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists():p=json.loads(RESULT_PATH.read_text()).get(mode,{}).get("best_params",{});return tuple(int(p[k]) for k in keys)
    return tuple(int(CONFIG["model"][k]) for k in keys)
def main(device_name=None,mode=None,epochs=None):
    seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);device=device_from(device_name or CONFIG["runtime"]["device"]);mode=mode or CONFIG["model"]["constraint_mode"];alpha=float(CONFIG["problem"]["diffusivity"]);g=torch.Generator(device=device).manual_seed(seed);model=DiffusionPIKAN(*architecture(mode),mode,device,seed).to(device);s=CONFIG["sampling"];inside=sample(int(s["interior_points"]),device,g);bc=boundary(int(s["boundary_points"]),device,g);ic=initial(int(s["initial_points"]),device);cfg=dict(CONFIG["training"]);cfg["epochs"]=epochs or cfg["epochs"];history=train(model,lambda:physics_loss(model,inside,bc,ic,alpha,CONFIG["loss_weights"]),cfg);validation=sample(int(s["validation_points"]),device,g);error=float(relative_l2(model,validation,alpha));print(f"Validation relative L2 error: {error:.6e}");save(model,history,error,CONFIG,alpha,"pikan")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("hard","soft"));p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.constraint,a.epochs)
