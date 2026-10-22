"""B-spline inverse identification of Kirchhoff plate flexural rigidity."""

from __future__ import annotations

import argparse
import json
import math

import torch

from _common import (
    ROOT, device_from, exact_solution, initialize_positive_parameter,
    load_value, observations, positive_parameter, problem, relative_l2,
    sample_points, save_results, set_seed, train_inverse,
)

CONFIG=json.loads((ROOT/"bspline_config.json").read_text(encoding="utf-8"))
RESULT_PATH=ROOT/CONFIG["output"]["tuned_hyperparameters"]


def open_uniform_knots(n_basis,degree,start,end,device):
    if n_basis<=degree: raise ValueError("n_basis must be greater than degree")
    count=n_basis-degree-1
    internal=torch.linspace(start,end,count+2,device=device)[1:-1] if count else torch.empty(0,device=device)
    return torch.cat((torch.full((degree+1,),start,device=device),internal,torch.full((degree+1,),end,device=device)))


def basis_matrix(values,knots,degree):
    values=values.reshape(-1);basis=((values[:,None]>=knots[:-1])&(values[:,None]<knots[1:])).to(values.dtype);right=values==knots[-1]
    if right.any(): basis[right]=0;basis[right,-1]=1
    for current in range(1,degree+1):
        columns=knots.numel()-current-1;updated=torch.zeros((values.numel(),columns),device=values.device,dtype=values.dtype)
        for index in range(columns):
            left=knots[index+current]-knots[index];other=knots[index+current+1]-knots[index+1]
            if left>0: updated[:,index]+=(values-knots[index])/left*basis[:,index]
            if other>0: updated[:,index]+=(knots[index+current+1]-values)/other*basis[:,index+1]
        basis=updated
    if right.any(): basis[right]=0;basis[right,-1]=1
    return basis


def basis_derivative_matrix(values,knots,degree,order):
    values=values.reshape(-1);n_basis=knots.numel()-degree-1
    if order==0:return basis_matrix(values,knots,degree)
    if order>degree:return torch.zeros((values.numel(),n_basis),device=values.device,dtype=values.dtype)
    right=values==knots[-1]
    if right.any():values=values.clone();values[right]=torch.nextafter(knots[-1],knots[0])
    lower=basis_derivative_matrix(values,knots,degree-1,order-1);result=torch.zeros((values.numel(),n_basis),device=values.device,dtype=values.dtype)
    for index in range(n_basis):
        left=knots[index+degree]-knots[index];other=knots[index+degree+1]-knots[index+1]
        if left>0:result[:,index]+=degree/left*lower[:,index]
        if other>0:result[:,index]-=degree/other*lower[:,index+1]
    return result


class KirchhoffInverseBSpline(torch.nn.Module):
    """Tensor-product spline with exact w=0 and M_nn=0 constraints."""
    constraint_mode="hard"

    def __init__(self,n_basis_x,n_basis_y,degree,values,inverse,device):
        super().__init__()
        if degree<4:raise ValueError("The fourth-order PDE requires degree >= 4")
        if min(n_basis_x,n_basis_y)<6:raise ValueError("At least six basis functions are required")
        self.degree=degree;self.values=values;self.minimum_d=float(inverse["minimum_d"])
        self.register_buffer("knots_x",open_uniform_knots(n_basis_x,degree,0,values["length_x"],device));self.register_buffer("knots_y",open_uniform_knots(n_basis_y,degree,0,values["length_y"],device))
        self.register_buffer("constraint_x",self._transform(self.knots_x,degree,n_basis_x,values["length_x"],device));self.register_buffer("constraint_y",self._transform(self.knots_y,degree,n_basis_y,values["length_y"],device))
        self.independent_coefficients=torch.nn.Parameter(1e-3*torch.randn(n_basis_x-4,n_basis_y-4,device=device));self.raw_d=initialize_positive_parameter(inverse["initial_d"],self.minimum_d,device)

    @staticmethod
    def _transform(knots,degree,n_basis,length,device):
        second=basis_derivative_matrix(torch.tensor((0.,length),device=device),knots,degree,2);transform=torch.zeros(n_basis,n_basis-4,device=device)
        transform[1,0]=1;transform[2,0]=-second[0,1]/second[0,2];transform[-2,-1]=1;transform[-3,-1]=-second[1,-2]/second[1,-3]
        for index in range(3,n_basis-3):transform[index,index-2]=1
        return transform

    def flexural_rigidity(self):return positive_parameter(self.raw_d,self.minimum_d)
    def coefficients(self):return self.constraint_x@self.independent_coefficients@self.constraint_y.T
    def operators(self,points,max_order=4):
        return {"x":tuple(basis_derivative_matrix(points[:,0],self.knots_x,self.degree,k) for k in range(max_order+1)),"y":tuple(basis_derivative_matrix(points[:,1],self.knots_y,self.degree,k) for k in range(max_order+1))}
    def evaluate(self,operators,dx=0,dy=0):return torch.einsum("ni,ij,nj->n",operators["x"][dx],self.coefficients(),operators["y"][dy])
    def forward(self,points):return self.evaluate(self.operators(points,0))


def inverse_loss(model,interior_ops,interior,sensor_ops,sensor_values,values,weights):
    bih=model.evaluate(interior_ops,4,0)+2*model.evaluate(interior_ops,2,2)+model.evaluate(interior_ops,0,4)
    pde=((model.flexural_rigidity()*bih-load_value(interior,values))/values["load"]).square().mean()
    center=torch.tensor([[values["length_x"]/2,values["length_y"]/2]],device=interior.device);scale=exact_solution(center,values)[0]
    data=((model.evaluate(sensor_ops)-sensor_values)/scale).square().mean()
    return float(weights["pde"])*pde+float(weights["data"])*data


def load_architecture():
    payload=json.loads(RESULT_PATH.read_text(encoding="utf-8"));params=payload.get("best_params",payload)
    return int(params["n_basis_x"]),int(params["n_basis_y"]),int(params["degree"])


def main(device_name=None,epochs=None,use_tuned=None):
    seed=int(CONFIG["reproducibility"]["seed"]);set_seed(seed);device=device_from(device_name or CONFIG["runtime"]["device"]);values=problem(CONFIG)
    load_tuned=(CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists()) if use_tuned is None else use_tuned
    architecture=load_architecture() if load_tuned else tuple(int(CONFIG["model"][k]) for k in ("n_basis_x","n_basis_y","degree"))
    generator=torch.Generator(device=device).manual_seed(seed);sampling=CONFIG["sampling"]
    interior=sample_points(int(sampling["interior_points"]),values,device,generator);sensors=observations(int(sampling["sensor_points"]),float(CONFIG["inverse"]["noise_std"]),values,device,generator)
    model=KirchhoffInverseBSpline(*architecture,values,CONFIG["inverse"],device).to(device);interior_ops=model.operators(interior,4);sensor_ops=model.operators(sensors[0],0)
    loss_fn=lambda:inverse_loss(model,interior_ops,interior,sensor_ops,sensors[1],values,CONFIG["loss_weights"]);training=dict(CONFIG["training"])
    if epochs is not None:training["epochs"]=epochs
    print(f"B-spline inverse: architecture={architecture}, exact simply-supported constraints, device={device}")
    history,d_history=train_inverse(model,loss_fn,training);validation=sample_points(int(sampling["validation_points"]),values,device,generator);error=float(relative_l2(model,validation,values));estimated=float(model.flexural_rigidity().detach())
    print(f"Estimated D: {estimated:.8f} (true {values['true_d']:.8f})");print(f"Relative D error: {abs(estimated-values['true_d'])/values['true_d']:.6e}");print(f"Validation relative L2 error: {error:.6e}")
    save_results(model,history,d_history,error,CONFIG,values,"bspline",sensors)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--device",choices=("cpu","gpu"),default=None);parser.add_argument("--epochs",type=int,default=None)
    group=parser.add_mutually_exclusive_group();group.add_argument("--use-tuned",dest="use_tuned",action="store_true");group.add_argument("--no-use-tuned",dest="use_tuned",action="store_false");parser.set_defaults(use_tuned=None);args=parser.parse_args();main(args.device,args.epochs,args.use_tuned)
