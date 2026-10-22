"""Tune inverse wave B-splines."""
import argparse,json,optuna,torch
from Bspline_2D import CONFIG,RESULT_PATH,InverseWaveBSpline,objective
from _common import device_from,observations,relative_l2,sample,set_seed,train
def main(device_name=None,trials=None,epochs=None):
 d=device_from(device_name or CONFIG["runtime"]["device"]);q=CONFIG["tuning"];seed=int(CONFIG["reproducibility"]["seed"]);true=float(CONFIG["problem"]["true_wave_speed"]);g=torch.Generator(device=d).manual_seed(seed);inside=sample(q["interior_points"],d,g);data=observations(q["sensor_points"],CONFIG["inverse"]["noise_std"],true,d,g);valid=sample(q["validation_points"],d,g)
 def fn(t):
  set_seed(seed);p=t.suggest_int("degree",q["degree"]["min"],q["degree"]["max"]);nx=t.suggest_int("n_basis_x",max(p+1,q["n_basis_x"]["min"]),q["n_basis_x"]["max"]);nt=t.suggest_int("n_basis_t",max(p+1,q["n_basis_t"]["min"]),q["n_basis_t"]["max"]);m=InverseWaveBSpline(nx,nt,p,CONFIG["inverse"],d);io=m.ops(inside);so=m.ops(data[0],0,0);cfg=dict(CONFIG["training"]);cfg["epochs"]=int(epochs or q["epochs"]);cfg["print_every"]=0;train(m,lambda:objective(m,io,so,data[1],CONFIG["loss_weights"]),cfg);field=float(relative_l2(m,valid,true));parameter=abs(float(m.wave_speed())-true)/true;t.set_user_attr("field_relative_l2",field);t.set_user_attr("parameter_relative_error",parameter);return field+parameter
 s=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed));s.optimize(fn,n_trials=int(trials or q["trials"]));RESULT_PATH.write_text(json.dumps({"best_value":s.best_value,"best_params":s.best_params,"best_metrics":s.best_trial.user_attrs},indent=2));print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.trials,a.epochs)
