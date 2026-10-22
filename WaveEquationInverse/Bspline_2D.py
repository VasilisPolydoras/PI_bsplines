"""Hard-constrained B-spline inverse wave-speed solver."""
import argparse,json,torch
from _common import ROOT,device_from,observations,positive,raw_positive,relative_l2,sample,save,set_seed,train
CONFIG=json.loads((ROOT/"bspline_config.json").read_text());RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]
def knots(n,p,d):
 q=n-p-1;i=torch.linspace(0,1,q+2,device=d)[1:-1] if q else torch.empty(0,device=d);return torch.cat((torch.zeros(p+1,device=d),i,torch.ones(p+1,device=d)))
def basis(x,k,p):
 x=x.reshape(-1);b=((x[:,None]>=k[:-1])&(x[:,None]<k[1:])).to(x.dtype);r=x==k[-1]
 if r.any():b[r]=0;b[r,-1]=1
 for d in range(1,p+1):
  o=torch.zeros((x.numel(),k.numel()-d-1),device=x.device)
  for j in range(o.shape[1]):
   a=k[j+d]-k[j];c=k[j+d+1]-k[j+1]
   if a>0:o[:,j]+=(x-k[j])/a*b[:,j]
   if c>0:o[:,j]+=(k[j+d+1]-x)/c*b[:,j+1]
  b=o
 if r.any():b[r]=0;b[r,-1]=1
 return b
def dbasis(x,k,p,n):
 x=x.reshape(-1);size=k.numel()-p-1
 if n==0:return basis(x,k,p)
 if n>p:return torch.zeros((x.numel(),size),device=x.device)
 r=x==k[-1]
 if r.any():x=x.clone();x[r]=torch.nextafter(k[-1],k[0])
 low=dbasis(x,k,p-1,n-1);o=torch.zeros((x.numel(),size),device=x.device)
 for j in range(size):
  a=k[j+p]-k[j];c=k[j+p+1]-k[j+1]
  if a>0:o[:,j]+=p/a*low[:,j]
  if c>0:o[:,j]-=p/c*low[:,j+1]
 return o
def greville(k,p):return torch.stack([k[i+1:i+p+1].mean() for i in range(k.numel()-p-1)])
class InverseWaveBSpline(torch.nn.Module):
 constraint_mode="hard"
 def __init__(self,nx,nt,p,inv,d):
  super().__init__();self.degree=p;self.minimum=float(inv["minimum_wave_speed"]);self.register_buffer("kx",knots(nx,p,d));self.register_buffer("kt",knots(nt,p,d));g=greville(self.kx,p);v=torch.linalg.lstsq(basis(g,self.kx,p),torch.sin(torch.pi*g)).solution;v[0]=0;v[-1]=0;self.register_buffer("initial_trace",v);dt0=dbasis(torch.zeros(1,device=d),self.kt,p,1)[0];self.register_buffer("velocity_ratio",-dt0[0]/dt0[1]);self.free=torch.nn.Parameter(torch.zeros(nt-2,nx-2,device=d));self.raw_c=raw_positive(inv["initial_wave_speed"],self.minimum,d)
 def wave_speed(self):return positive(self.raw_c,self.minimum)
 def coefficients(self):
  c=torch.zeros((self.kt.numel()-self.degree-1,self.kx.numel()-self.degree-1),device=self.free.device);c[0]=self.initial_trace;c[1]=self.velocity_ratio*self.initial_trace;c[2:,1:-1]=self.free;return c
 def ops(self,z,mx=2,mt=2):return {"x":tuple(dbasis(z[:,0],self.kx,self.degree,i) for i in range(mx+1)),"t":tuple(dbasis(z[:,1],self.kt,self.degree,i) for i in range(mt+1))}
 def eval(self,o,dx=0,dt=0):return torch.einsum("nt,tx,nx->n",o["t"][dt],self.coefficients(),o["x"][dx])
 def forward(self,z):return self.eval(self.ops(z,0,0))
def objective(m,io,so,values,w):return float(w["pde"])*(m.eval(io,0,2)-m.wave_speed().square()*m.eval(io,2,0)).square().mean()+float(w["data"])*(m.eval(so)-values).square().mean()
def arch():
 if CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists():p=json.loads(RESULT_PATH.read_text()).get("best_params",{});return int(p["n_basis_x"]),int(p["n_basis_t"]),int(p["degree"])
 return tuple(int(CONFIG["model"][k]) for k in ("n_basis_x","n_basis_t","degree"))
def main(device_name=None,epochs=None):
 seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);d=device_from(device_name or CONFIG["runtime"]["device"]);true=float(CONFIG["problem"]["true_wave_speed"]);g=torch.Generator(device=d).manual_seed(seed);s=CONFIG["sampling"];inside=sample(s["interior_points"],d,g);data=observations(s["sensor_points"],CONFIG["inverse"]["noise_std"],true,d,g);m=InverseWaveBSpline(*arch(),CONFIG["inverse"],d);io=m.ops(inside);so=m.ops(data[0],0,0);cfg=dict(CONFIG["training"]);cfg["epochs"]=epochs or cfg["epochs"];h,ch=train(m,lambda:objective(m,io,so,data[1],CONFIG["loss_weights"]),cfg);valid=sample(s["validation_points"],d,g);e=float(relative_l2(m,valid,true));est=float(m.wave_speed());print(f"Estimated wave speed: {est:.8f} (true {true:.8f})");print(f"Validation relative L2: {e:.6e}");save(m,h,ch,e,CONFIG,true,"bspline",data)
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.epochs)
