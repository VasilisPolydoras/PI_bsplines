"""Physics-informed tensor-product B-spline diffusion solver."""
import argparse,json
import torch
from _common import ROOT,device_from,exact,relative_l2,sample,save,set_seed,train
CONFIG=json.loads((ROOT/"bspline_config.json").read_text());RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]
def knots(n,p,device):
    q=n-p-1;inside=torch.linspace(0,1,q+2,device=device)[1:-1] if q else torch.empty(0,device=device);return torch.cat((torch.zeros(p+1,device=device),inside,torch.ones(p+1,device=device)))
def basis(x,k,p):
    x=x.reshape(-1);b=((x[:,None]>=k[:-1])&(x[:,None]<k[1:])).to(x.dtype);right=x==k[-1]
    if right.any():b[right]=0;b[right,-1]=1
    for d in range(1,p+1):
        out=torch.zeros((x.numel(),k.numel()-d-1),device=x.device)
        for i in range(out.shape[1]):
            a=k[i+d]-k[i];c=k[i+d+1]-k[i+1]
            if a>0:out[:,i]+=(x-k[i])/a*b[:,i]
            if c>0:out[:,i]+=(k[i+d+1]-x)/c*b[:,i+1]
        b=out
    if right.any():b[right]=0;b[right,-1]=1
    return b
def dbasis(x,k,p,o):
    x=x.reshape(-1);n=k.numel()-p-1
    if o==0:return basis(x,k,p)
    if o>p:return torch.zeros((x.numel(),n),device=x.device)
    right=x==k[-1]
    if right.any():x=x.clone();x[right]=torch.nextafter(k[-1],k[0])
    low=dbasis(x,k,p-1,o-1);out=torch.zeros((x.numel(),n),device=x.device)
    for i in range(n):
        a=k[i+p]-k[i];c=k[i+p+1]-k[i+1]
        if a>0:out[:,i]+=p/a*low[:,i]
        if c>0:out[:,i]-=p/c*low[:,i+1]
    return out
def greville(k,p):return torch.stack([k[i+1:i+p+1].mean() for i in range(k.numel()-p-1)])
class DiffusionBSpline(torch.nn.Module):
    constraint_mode="hard"
    def __init__(self,nx,nt,degree,device):
        super().__init__();self.degree=degree;self.register_buffer("kx",knots(nx,degree,device));self.register_buffer("kt",knots(nt,degree,device));g=greville(self.kx,degree);trace=torch.linalg.lstsq(basis(g,self.kx,degree),torch.sin(2*torch.pi*g)).solution;trace[0]=0;trace[-1]=0;self.register_buffer("initial_trace",trace);self.free=torch.nn.Parameter(1e-2*torch.randn(nx-2,nt-1,device=device))
    def coefficients(self):
        c=torch.zeros((self.kx.numel()-self.degree-1,self.kt.numel()-self.degree-1),device=self.free.device);c[:,0]=self.initial_trace;c[1:-1,1:]=self.free;return c
    def ops(self,z,max_x=2,max_t=1):return {"x":tuple(dbasis(z[:,0],self.kx,self.degree,i) for i in range(max_x+1)),"t":tuple(dbasis(z[:,1],self.kt,self.degree,i) for i in range(max_t+1))}
    def eval(self,o,dx=0,dt=0):return torch.einsum("ni,ij,nj->n",o["x"][dx],self.coefficients(),o["t"][dt])
    def forward(self,z):return self.eval(self.ops(z,0,0))
def loss(model,o,alpha):return (model.eval(o,0,1)-alpha*model.eval(o,2,0)).square().mean()
def architecture():
    if CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists():p=json.loads(RESULT_PATH.read_text()).get("best_params",{});return int(p["n_basis_x"]),int(p["n_basis_t"]),int(p["degree"])
    return tuple(int(CONFIG["model"][k]) for k in ("n_basis_x","n_basis_t","degree"))
def main(device_name=None,epochs=None):
    seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);device=device_from(device_name or CONFIG["runtime"]["device"]);alpha=float(CONFIG["problem"]["diffusivity"]);g=torch.Generator(device=device).manual_seed(seed);model=DiffusionBSpline(*architecture(),device);points=sample(int(CONFIG["sampling"]["interior_points"]),device,g);o=model.ops(points);cfg=dict(CONFIG["training"]);cfg["epochs"]=epochs or cfg["epochs"];history=train(model,lambda:loss(model,o,alpha),cfg);validation=sample(int(CONFIG["sampling"]["validation_points"]),device,g);error=float(relative_l2(model,validation,alpha));print(f"Validation relative L2 error: {error:.6e}");save(model,history,error,CONFIG,alpha,"bspline")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.epochs)
