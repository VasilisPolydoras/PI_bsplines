"""Tune Kirchhoff-plate B-splines with a seeded Optuna study."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import optuna, torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from hyperparameter_optimization import run_tpe_study
from training import train
from Kirchoff.Bspline_2D import CONFIG,LEARNING_RATE,RESULT_PATH,SEED,KirchhoffBSpline,exact_solution,pde_loss,precompute_operators,sample_interior,select_device,set_seed

def main(device_name=None,trials=None,epochs=None):
    tuning=CONFIG["tuning"]; device=select_device(device_name or CONFIG["runtime"]["device"]); generator=torch.Generator(device=device).manual_seed(SEED)
    interior=sample_interior(int(tuning["training_points"]),device,generator); validation=sample_interior(int(tuning["validation_points"]),device,generator)
    def objective(trial:optuna.Trial):
        set_seed(SEED); degree=trial.suggest_int("degree",int(tuning["degree"]["min"]),int(tuning["degree"]["max"])); n_basis=trial.suggest_int("n_basis",max(degree+1,int(tuning["n_basis"]["min"])),int(tuning["n_basis"]["max"])); model=KirchhoffBSpline(n_basis,n_basis,degree,device); operators=precompute_operators(model,interior); optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=tuning["scheduler"]["factor"],patience=tuning["scheduler"]["patience"],min_lr=tuning["scheduler"]["min_learning_rate"]); stopping=tuning["early_stopping"]
        train(model.parameters(),lambda:pde_loss(model,operators,interior),optimizer,epochs=int(epochs or tuning["epochs"]),scheduler=scheduler,early_stopping_patience=stopping["patience"],min_delta=stopping["min_delta"],print_every=0)
        with torch.no_grad(): prediction=model(validation).squeeze(-1); reference=exact_solution(validation)
        return float(torch.linalg.vector_norm(prediction-reference)/torch.linalg.vector_norm(reference))
    study=run_tpe_study(objective,n_trials=int(trials or tuning["trials"]),seed=SEED,direction="minimize",result_path=RESULT_PATH); print(f"Best validation relative L2 error: {study.best_value:.6e}"); print(f"Best hyperparameters: {study.best_params}"); print(f"Saved: {RESULT_PATH}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--device",choices=("cpu","gpu")); p.add_argument("--trials",type=int); p.add_argument("--epochs",type=int); a=p.parse_args(); main(a.device,a.trials,a.epochs)
