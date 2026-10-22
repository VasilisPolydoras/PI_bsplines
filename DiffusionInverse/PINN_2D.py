"""PINN inverse identification of diffusion coefficient."""
import argparse,json,torch
from _common import ROOT,boundary,device_from,initial,loss,observations,positive,raw_positive,relative_l2,sample,save,set_seed,train
CONFIG=json.loads((ROOT/"pinn_config.json").read_text());RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]
class InversePINN(torch.nn.Module):
 def __init__(self,layers,neurons,mode,inv,device):
  super().__init__();self.constraint_mode=mode;self.minimum=float(inv["minimum_diffusivity"]);self.raw_alpha=raw_positive(inv["initial_diffusivity"],self.minimum,device);m=[];w=2
  for _ in range(layers):q=torch.nn.Linear(w,neurons);torch.nn.init.xavier_normal_(q.weight);torch.nn.init.zeros_(q.bias);m.extend((q,torch.nn.Tanh()));w=neurons
  m.append(torch.nn.Linear(w,1));self.net=torch.nn.Sequential(*m)
 def diffusivity(self):return positive(self.raw_alpha,self.minimum)
 def forward(self,z):
  x,t=z[:,0],z[:,1];r=self.net(torch.stack((2*x-1,2*t-1),1)).squeeze(-1);return torch.sin(2*torch.pi*x)+t*x*(1-x)*r if self.constraint_mode=="hard" else r
def arch(mode):
 if CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists():p=json.loads(RESULT_PATH.read_text()).get(mode,{}).get("best_params",{});return int(p["n_layers"]),int(p["n_neurons"])
 return int(CONFIG["model"]["n_layers"]),int(CONFIG["model"]["n_neurons"])
def main(device_name=None,mode=None,epochs=None):
 seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);d=device_from(device_name or CONFIG["runtime"]["device"]);mode=mode or CONFIG["model"]["constraint_mode"];true=float(CONFIG["problem"]["true_diffusivity"]);g=torch.Generator(device=d).manual_seed(seed);s=CONFIG["sampling"];inside=sample(s["interior_points"],d,g);data=observations(s["sensor_points"],CONFIG["inverse"]["noise_std"],true,d,g);bc=boundary(s["boundary_points"],d,g);ic=initial(s["initial_points"],d);model=InversePINN(*arch(mode),mode,CONFIG["inverse"],d).to(d);cfg=dict(CONFIG["training"]);cfg["epochs"]=epochs or cfg["epochs"];h,ah=train(model,lambda:loss(model,inside,*data,bc,ic,true,CONFIG["loss_weights"]),cfg);valid=sample(s["validation_points"],d,g);e=float(relative_l2(model,valid,true));est=float(model.diffusivity());print(f"Estimated diffusivity: {est:.8f} (true {true:.8f})");print(f"Validation relative L2: {e:.6e}");save(model,h,ah,e,CONFIG,true,"pinn",data)
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("hard","soft"));p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.constraint,a.epochs)
