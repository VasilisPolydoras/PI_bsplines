"""Tune the Kirchhoff-plate PINN with a seeded Optuna study."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import optuna,torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from hyperparameter_optimization import run_tpe_study
from training import train
from Kirchoff.PINN_2D import CONFIG,LEARNING_RATE,RESULT_PATH,SEED,KirchhoffPINN,pinn_loss,relative_l2_error,sample_boundary,sample_interior,select_device,set_seed

def main(device_name=None,constraint_mode=None,trials=None,epochs=None):
    tuning=CONFIG["tuning"]; mode=constraint_mode or CONFIG["model"]["constraint_mode"]; device=select_device(device_name or CONFIG["runtime"]["device"]); generator=torch.Generator(device=device).manual_seed(SEED); interior=sample_interior(int(tuning["training_points"]),device,generator); validation=sample_interior(int(tuning["validation_points"]),device,generator); boundaries=sample_boundary(int(tuning["boundary_points_per_edge"]),device)
    def objective(trial:optuna.Trial):
        set_seed(SEED); model=KirchhoffPINN(trial.suggest_int("n_layers",tuning["n_layers"]["min"],tuning["n_layers"]["max"]),trial.suggest_int("n_neurons",tuning["n_neurons"]["min"],tuning["n_neurons"]["max"],step=tuning["n_neurons"]["step"]),mode).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=tuning["scheduler"]["factor"],patience=tuning["scheduler"]["patience"],min_lr=tuning["scheduler"]["min_learning_rate"]); stopping=tuning["early_stopping"]; train(model.parameters(),lambda:pinn_loss(model,interior,boundaries),optimizer,epochs=int(epochs or tuning["epochs"]),scheduler=scheduler,early_stopping_patience=stopping["patience"],min_delta=stopping["min_delta"],print_every=0); return float(relative_l2_error(model,validation))
    study=run_tpe_study(objective,n_trials=int(trials or tuning["trials"]),seed=SEED,direction="minimize",result_path=RESULT_PATH); print(f"Constraint mode: {mode}"); print(f"Best validation relative L2 error: {study.best_value:.6e}"); print(f"Best hyperparameters: {study.best_params}"); print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--device",choices=("cpu","gpu")); p.add_argument("--constraint",choices=("hard","soft")); p.add_argument("--trials",type=int); p.add_argument("--epochs",type=int); a=p.parse_args(); main(a.device,a.constraint,a.trials,a.epochs)
