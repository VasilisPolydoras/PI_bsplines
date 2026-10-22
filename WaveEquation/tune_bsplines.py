"""Tune forward wave B-splines."""
import argparse,json,optuna,torch
from Bspline_2D import CONFIG,RESULT_PATH,WaveBSpline,objective
from _common import device_from,relative_l2,sample,set_seed,train
def main(device_name=None,trials=None,epochs=None):
 d=device_from(device_name or CONFIG["runtime"]["device"]);q=CONFIG["tuning"];seed=int(CONFIG["reproducibility"]["seed"]);c=float(CONFIG["problem"]["wave_speed"]);g=torch.Generator(device=d).manual_seed(seed);inside=sample(q["interior_points"],d,g);valid=sample(q["validation_points"],d,g)
 def fn(t):
  set_seed(seed);p=t.suggest_int("degree",q["degree"]["min"],q["degree"]["max"]);nx=t.suggest_int("n_basis_x",max(p+1,q["n_basis_x"]["min"]),q["n_basis_x"]["max"]);nt=t.suggest_int("n_basis_t",max(p+1,q["n_basis_t"]["min"]),q["n_basis_t"]["max"]);m=WaveBSpline(nx,nt,p,d);o=m.ops(inside);cfg=dict(CONFIG["training"]);cfg["epochs"]=int(epochs or q["epochs"]);cfg["print_every"]=0;train(m,lambda:objective(m,o,c),cfg);return float(relative_l2(m,valid,c))
 s=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed));s.optimize(fn,n_trials=int(trials or q["trials"]));RESULT_PATH.write_text(json.dumps({"best_value":s.best_value,"best_params":s.best_params},indent=2));print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.trials,a.epochs)
