"""Tune the inverse-diffusion PINN with Optuna TPE."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import optuna,torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from hyperparameter_optimization import run_tpe_study
from training import train
from DiffusionInverse.PINN_2D import CONFIG,RESULT_PATH,SEED,TRUE_DIFFUSIVITY,InverseDiffusionPINN,inverse_loss,relative_l2_error,sample_constraints,sample_interior,sample_observations,select_device,set_seed
def main(device_name=None,constraint_mode=None,trials=None,epochs=None):
 q=CONFIG["tuning"];mode=constraint_mode or CONFIG["model"]["constraint_mode"];d=select_device(device_name or CONFIG["runtime"]["device"]);g=torch.Generator(device=d).manual_seed(SEED);inside=sample_interior(q["training_points"],d,generator=g);sensors=sample_observations(q["sensor_points"],CONFIG["inverse"]["noise_std"],TRUE_DIFFUSIVITY,d,g);valid=sample_interior(q["validation_points"],d,generator=g);constraints=sample_constraints(q["boundary_points_per_edge"],q["initial_points"],d);training=CONFIG["training"]
 def objective(trial:optuna.Trial):
  set_seed(SEED);m=InverseDiffusionPINN(trial.suggest_int("n_layers",q["n_layers"]["min"],q["n_layers"]["max"]),trial.suggest_int("n_neurons",q["n_neurons"]["min"],q["n_neurons"]["max"],step=q["n_neurons"]["step"]),mode,CONFIG["inverse"],d).to(d);optimizer=torch.optim.Adam(m.parameters(),lr=training["learning_rate"]);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=q["scheduler"]["factor"],patience=q["scheduler"]["patience"],min_lr=q["scheduler"]["min_learning_rate"]);train(m.parameters(),lambda:inverse_loss(m,inside,sensors[0],sensors[1],constraints),optimizer,epochs=int(epochs or q["epochs"]),scheduler=scheduler,early_stopping_patience=q["early_stopping"]["patience"],min_delta=q["early_stopping"]["min_delta"],print_every=0);field=float(relative_l2_error(m,valid));parameter=abs(float(m.diffusivity())-TRUE_DIFFUSIVITY)/TRUE_DIFFUSIVITY;trial.set_user_attr("field_relative_l2",field);trial.set_user_attr("parameter_relative_error",parameter);return field+parameter
 temp=RESULT_PATH.with_name(f".{RESULT_PATH.stem}_{mode}.json");study=run_tpe_study(objective,n_trials=int(trials or q["trials"]),seed=SEED,direction="minimize",result_path=temp);result={"best_value":study.best_value,"best_params":study.best_params,"best_metrics":study.best_trial.user_attrs};payload=json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {};payload[mode]=result;RESULT_PATH.write_text(json.dumps(payload,indent=2),encoding="utf-8");temp.unlink(missing_ok=True);print(f"Best objective: {study.best_value:.6e}\nBest hyperparameters: {study.best_params}\nSaved: {RESULT_PATH}")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("soft","hard"));p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.constraint,a.trials,a.epochs)
