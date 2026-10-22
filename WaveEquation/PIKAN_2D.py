"""PIKAN wave solver with hard or soft initial/boundary constraints."""
import argparse,json,torch
try:from kan import KAN
except ImportError:KAN=None
from _common import ROOT,boundary,device_from,initial,loss,relative_l2,sample,save,set_seed,train
CONFIG=json.loads((ROOT/"pikan_config.json").read_text());RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]
class WavePIKAN(torch.nn.Module):
 def __init__(self,layers,neurons,grid,k,mode,d,seed):
  super().__init__();
  if KAN is None:raise ImportError("PIKAN requires pykan (import name 'kan').")
  self.constraint_mode=mode;self.net=KAN(width=[2]+[neurons]*layers+[1],grid=grid,k=k,seed=seed,device=d,auto_save=False,save_act=False,symbolic_enabled=False)
 def forward(self,z):
  x,t=z[:,0],z[:,1];r=self.net(torch.stack((2*x-1,2*t-1),1)).squeeze(-1);return torch.sin(torch.pi*x)+t.square()*x*(1-x)*r if self.constraint_mode=="hard" else r
def arch(mode):
 keys=("n_layers","n_neurons","grid","spline_order")
 if CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists():p=json.loads(RESULT_PATH.read_text()).get(mode,{}).get("best_params",{});return tuple(int(p[k]) for k in keys)
 return tuple(int(CONFIG["model"][k]) for k in keys)
def main(device_name=None,mode=None,epochs=None):
 seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);d=device_from(device_name or CONFIG["runtime"]["device"]);mode=mode or CONFIG["model"]["constraint_mode"];c=float(CONFIG["problem"]["wave_speed"]);g=torch.Generator(device=d).manual_seed(seed);m=WavePIKAN(*arch(mode),mode,d,seed).to(d);s=CONFIG["sampling"];inside=sample(s["interior_points"],d,g);bc=boundary(s["boundary_points"],d,g);ic=initial(s["initial_points"],d);cfg=dict(CONFIG["training"]);cfg["epochs"]=epochs or cfg["epochs"];h=train(m,lambda:loss(m,inside,bc,ic,c,CONFIG["loss_weights"]),cfg);valid=sample(s["validation_points"],d,g);e=float(relative_l2(m,valid,c));print(f"Validation relative L2: {e:.6e}");save(m,h,e,CONFIG,c,"pikan")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("hard","soft"));p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.constraint,a.epochs)
