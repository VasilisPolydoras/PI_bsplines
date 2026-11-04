"""Repeat one Kirchhoff solver with fixed samples and varied initialization."""
from __future__ import annotations
import argparse,contextlib,gc,io,math,sys
from pathlib import Path
import numpy as np,torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from Kirchoff.Bspline_2D import DEFAULT_DEVICE,SEED
SCRIPT_DIR=Path(__file__).resolve().parent; DEFAULT_RUNS=10; DEFAULT_INITIALIZATION_SEED_START=32

def confidence_interval_95(values):
    if values.size<2: raise ValueError("At least two runs are required")
    try:
        from scipy.stats import t
        critical=float(t.ppf(.975,df=values.size-1))
    except ImportError:
        if values.size!=10: raise RuntimeError("SciPy is required when runs is not 10")
        critical=2.2621571628540993
    mean=float(values.mean()); half=critical*float(values.std(ddof=1))/math.sqrt(values.size); return mean-half,mean+half

def summarize(values): return {"mean":float(values.mean()),"std":float(values.std(ddof=1)),"ci95":np.asarray(confidence_interval_95(values))}

def solver_definition(name):
    if name=="bspline":
        from Kirchoff.Bspline_2D import EPOCHS,main
        return main,EPOCHS,"PI-TP-BsA",None
    if name=="pinn":
        from Kirchoff.PINN_2D import CONSTRAINT_MODE,EPOCHS,main
        return main,EPOCHS,"PINN",CONSTRAINT_MODE
    if name=="pikan":
        from Kirchoff.PIKAN_2D import CONSTRAINT_MODE,EPOCHS,main
        return main,EPOCHS,"PIKAN",CONSTRAINT_MODE
    raise ValueError("unknown solver")

def main(solver,runs=DEFAULT_RUNS,device_name=DEFAULT_DEVICE,sampling_seed=SEED,validation_seed=SEED,epochs=None,constraint_mode=None):
    if runs<2: raise ValueError("runs must be at least two")
    run_solver,configured_epochs,label,configured_constraint=solver_definition(solver)
    if solver=="bspline" and constraint_mode is not None: raise ValueError("--constraint applies only to PINN and PIKAN")
    selected=constraint_mode if constraint_mode is not None else configured_constraint; run_epochs=configured_epochs if epochs is None else epochs; seeds=np.arange(DEFAULT_INITIALIZATION_SEED_START,DEFAULT_INITIALIZATION_SEED_START+runs,dtype=np.int64); errors=np.empty(runs); times=np.empty(runs)
    for index,seed in enumerate(seeds):
        kwargs={"device_name":device_name,"epochs":run_epochs,"use_tuned":None,"initialization_seed":int(seed),"sampling_seed":sampling_seed,"validation_seed":validation_seed,"save_outputs":False,"verbose":False}
        if selected is not None: kwargs["constraint_mode"]=selected
        with contextlib.redirect_stdout(io.StringIO()): result=run_solver(**kwargs)
        errors[index]=result["validation_relative_l2_error"]; times[index]=result["training_time_seconds"]; method=label if selected is None else f"{label}-{selected}"; print(f"Run {index+1:02d}/{runs} | {method:12s} | seed={seed} | relative L2={errors[index]:.6e} | time={times[index]:.3f} s"); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    es,ts=summarize(errors),summarize(times); stem=solver if selected is None else f"{solver}_{selected}"; output=SCRIPT_DIR/"postprocessing"/f"{stem}_statistics.npz"; output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,solver=np.asarray(solver),constraint_mode=np.asarray("not_applicable" if selected is None else selected),initialization_seeds=seeds,sampling_seed=np.asarray(sampling_seed),validation_seed=np.asarray(validation_seed),epochs=np.asarray(run_epochs),l2_errors=errors,training_times_seconds=times,l2_mean=es["mean"],l2_std=es["std"],l2_ci95=es["ci95"],time_mean_seconds=ts["mean"],time_std_seconds=ts["std"],time_ci95_seconds=ts["ci95"])
    print(f"\nRelative L2: {es['mean']:.6e} +/- {es['std']:.6e}; 95% CI [{es['ci95'][0]:.6e}, {es['ci95'][1]:.6e}]"); print(f"Training time: {ts['mean']:.3f} +/- {ts['std']:.3f} s; 95% CI [{ts['ci95'][0]:.3f}, {ts['ci95'][1]:.3f}] s"); print(f"Saved: {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--solver",choices=("bspline","pinn","pikan"),required=True); p.add_argument("--runs",type=int,default=DEFAULT_RUNS); p.add_argument("--device",choices=("cpu","gpu"),default=DEFAULT_DEVICE); p.add_argument("--sampling-seed",type=int,default=SEED); p.add_argument("--validation-seed",type=int,default=SEED); p.add_argument("--epochs",type=int); p.add_argument("--constraint",choices=("soft","hard")); a=p.parse_args(); main(a.solver,a.runs,a.device,a.sampling_seed,a.validation_seed,a.epochs,a.constraint)
