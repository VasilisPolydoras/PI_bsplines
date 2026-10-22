"""Tune forward diffusion PINN."""
import argparse,json,optuna,torch
from PINN_2D import CONFIG,RESULT_PATH,DiffusionPINN
from _common import boundary,device_from,initial,physics_loss,relative_l2,sample,set_seed,train
def main(device_name=None,mode="hard",trials=None,epochs=None):
 d=device_from(device_name or CONFIG["runtime"]["device"]);q=CONFIG["tuning"];seed=int(CONFIG["reproducibility"]["seed"]);a=float(CONFIG["problem"]["diffusivity"]);g=torch.Generator(device=d).manual_seed(seed);inside=sample(q["interior_points"],d,g);bc=boundary(q["boundary_points"],d,g);ic=initial(q["initial_points"],d);valid=sample(q["validation_points"],d,g)
 def objective(t):
  set_seed(seed);m=DiffusionPINN(t.suggest_int("n_layers",q["n_layers"]["min"],q["n_layers"]["max"]),t.suggest_int("n_neurons",q["n_neurons"]["min"],q["n_neurons"]["max"],step=q["n_neurons"]["step"]),mode).to(d);c=dict(CONFIG["training"]);c["epochs"]=int(epochs or q["epochs"]);c["print_every"]=0;train(m,lambda:physics_loss(m,inside,bc,ic,a,CONFIG["loss_weights"]),c);return float(relative_l2(m,valid,a))
 s=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed));s.optimize(objective,n_trials=int(trials or q["trials"]));payload=json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {};payload[mode]={"best_value":s.best_value,"best_params":s.best_params};RESULT_PATH.write_text(json.dumps(payload,indent=2));print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("hard","soft"),default="hard");p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);x=p.parse_args();main(x.device,x.constraint,x.trials,x.epochs)
