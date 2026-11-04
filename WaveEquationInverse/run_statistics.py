"""Repeat one inverse-wave solver with fixed data and varied initialization."""
from __future__ import annotations
import argparse,contextlib,gc,io,math,sys
from pathlib import Path
import numpy as np
import torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from WaveEquationInverse.Bspline_2D import DEFAULT_DEVICE,SEED
ROOT=Path(__file__).resolve().parent;DEFAULT_RUNS=10;DEFAULT_SEED_START=32
def confidence_interval_95(values):
 try:
  from scipy.stats import t
  critical=float(t.ppf(0.975,df=values.size-1))
 except ImportError:
  if values.size!=10:raise RuntimeError("SciPy is required unless runs=10")
  critical=2.2621571628540993
 half=critical*float(values.std(ddof=1))/math.sqrt(values.size);mean=float(values.mean());return np.asarray((mean-half,mean+half))
def definition(name):
 if name=="bspline":
  from WaveEquationInverse.Bspline_2D import EPOCHS,main
  return main,EPOCHS,None
 if name=="pinn":
  from WaveEquationInverse.PINN_2D import CONSTRAINT_MODE,EPOCHS,main
  return main,EPOCHS,CONSTRAINT_MODE
 if name=="pikan":
  from WaveEquationInverse.PIKAN_2D import CONSTRAINT_MODE,EPOCHS,main
  return main,EPOCHS,CONSTRAINT_MODE
 raise ValueError("Unknown solver")
def main(solver,runs=DEFAULT_RUNS,device_name=DEFAULT_DEVICE,sampling_seed=SEED,validation_seed=SEED,epochs=None,constraint_mode=None):
 if runs<2:raise ValueError("runs must be at least two")
 run_solver,configured_epochs,configured_mode=definition(solver);mode=constraint_mode or configured_mode
 if solver=="bspline" and constraint_mode is not None:raise ValueError("--constraint applies only to PINN/PIKAN")
 seeds=np.arange(DEFAULT_SEED_START,DEFAULT_SEED_START+runs);field=np.empty(runs);parameter=np.empty(runs);times=np.empty(runs)
 for index,seed in enumerate(seeds):
  kwargs=dict(device_name=device_name,epochs=configured_epochs if epochs is None else epochs,use_tuned=None,initialization_seed=int(seed),sampling_seed=sampling_seed,validation_seed=validation_seed,save_outputs=False,verbose=False)
  if mode is not None:kwargs["constraint_mode"]=mode
  with contextlib.redirect_stdout(io.StringIO()):result=run_solver(**kwargs)
  field[index]=result["validation_relative_l2_error"];parameter[index]=result["parameter_relative_error"];times[index]=result["training_time_seconds"]
  print(f"Run {index+1:02d}/{runs} | seed={seed} | field L2={field[index]:.6e} | parameter={parameter[index]:.6e} | time={times[index]:.3f} s")
  gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
 stem=solver if mode is None else f"{solver}_{mode}";output=ROOT/"postprocessing"/f"{stem}_statistics.npz";output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output,solver=np.asarray(solver),constraint_mode=np.asarray("not_applicable" if mode is None else mode),initialization_seeds=seeds,sampling_seed=sampling_seed,validation_seed=validation_seed,field_l2_errors=field,parameter_relative_errors=parameter,training_times_seconds=times,field_mean=field.mean(),field_std=field.std(ddof=1),field_ci95=confidence_interval_95(field),parameter_mean=parameter.mean(),parameter_std=parameter.std(ddof=1),parameter_ci95=confidence_interval_95(parameter),time_mean_seconds=times.mean(),time_std_seconds=times.std(ddof=1),time_ci95_seconds=confidence_interval_95(times))
 print(f"\nField L2: {field.mean():.6e} +/- {field.std(ddof=1):.6e}; 95% CI {confidence_interval_95(field)}")
 print(f"Parameter error: {parameter.mean():.6e} +/- {parameter.std(ddof=1):.6e}; 95% CI {confidence_interval_95(parameter)}")
 print(f"Training time: {times.mean():.3f} +/- {times.std(ddof=1):.3f} s\nSaved: {output}")
if __name__=="__main__":
 parser=argparse.ArgumentParser(description="Inverse wave repeated-run statistics");parser.add_argument("--solver",choices=("bspline","pinn","pikan"),required=True);parser.add_argument("--runs",type=int,default=DEFAULT_RUNS);parser.add_argument("--device",choices=("cpu","gpu"),default=DEFAULT_DEVICE);parser.add_argument("--sampling-seed",type=int,default=SEED);parser.add_argument("--validation-seed",type=int,default=SEED);parser.add_argument("--epochs",type=int);parser.add_argument("--constraint",choices=("soft","hard"));args=parser.parse_args();main(args.solver,args.runs,args.device,args.sampling_seed,args.validation_seed,args.epochs,args.constraint)
