"""Optuna tuning for inverse Kirchhoff B-splines."""
import argparse,json
import optuna,torch
from Bspline_2D import CONFIG,RESULT_PATH,KirchhoffInverseBSpline,inverse_loss
from _common import device_from,observations,problem,relative_l2,sample_points,set_seed,train_inverse

def main(device_name=None,trials=None,epochs=None):
    tuning=CONFIG["tuning"];device=device_from(device_name or CONFIG["runtime"]["device"]);seed=int(CONFIG["reproducibility"]["seed"]);values=problem(CONFIG);g=torch.Generator(device=device).manual_seed(seed);interior=sample_points(int(tuning["interior_points"]),values,device,g);sensors=observations(int(tuning["sensor_points"]),float(CONFIG["inverse"]["noise_std"]),values,device,g);validation=sample_points(int(tuning["validation_points"]),values,device,g)
    def objective(trial):
        set_seed(seed);degree=trial.suggest_int("degree",tuning["degree"]["min"],tuning["degree"]["max"]);nx=trial.suggest_int("n_basis_x",max(degree+1,tuning["n_basis_x"]["min"]),tuning["n_basis_x"]["max"]);ny=trial.suggest_int("n_basis_y",max(degree+1,tuning["n_basis_y"]["min"]),tuning["n_basis_y"]["max"]);model=KirchhoffInverseBSpline(nx,ny,degree,values,CONFIG["inverse"],device);io=model.operators(interior,4);so=model.operators(sensors[0],0);fn=lambda:inverse_loss(model,io,interior,so,sensors[1],values,CONFIG["loss_weights"]);training=dict(CONFIG["training"]);training["epochs"]=int(epochs or tuning["epochs"]);training["print_every"]=0;train_inverse(model,fn,training);field=float(relative_l2(model,validation,values));parameter=abs(float(model.flexural_rigidity())-values["true_d"])/values["true_d"];trial.set_user_attr("field_relative_l2",field);trial.set_user_attr("parameter_relative_error",parameter);return field+parameter
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed));study.optimize(objective,n_trials=int(trials or tuning["trials"]));RESULT_PATH.write_text(json.dumps({"best_value":study.best_value,"best_params":study.best_params,"best_metrics":study.best_trial.user_attrs},indent=2));print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.trials,a.epochs)
