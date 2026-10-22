"""Optuna tuning for the inverse Kirchhoff PIKAN."""
import argparse,json
import optuna,torch
from PIKAN_2D import CONFIG,RESULT_PATH,KirchhoffInversePIKAN
from _common import device_from,loss,observations,problem,relative_l2,sample_boundary,sample_points,set_seed,train_inverse

def main(device_name=None,mode="hard",trials=None,epochs=None):
    tuning=CONFIG["tuning"];device=device_from(device_name or CONFIG["runtime"]["device"]);seed=int(CONFIG["reproducibility"]["seed"]);values=problem(CONFIG);g=torch.Generator(device=device).manual_seed(seed)
    interior=sample_points(int(tuning["interior_points"]),values,device,g);sensors=observations(int(tuning["sensor_points"]),float(CONFIG["inverse"]["noise_std"]),values,device,g);validation=sample_points(int(tuning["validation_points"]),values,device,g);boundaries=sample_boundary(int(tuning["boundary_points_per_edge"]),values,device)
    def objective(trial):
        set_seed(seed);architecture=(trial.suggest_int("n_layers",tuning["n_layers"]["min"],tuning["n_layers"]["max"]),trial.suggest_int("n_neurons",tuning["n_neurons"]["min"],tuning["n_neurons"]["max"],step=tuning["n_neurons"]["step"]),trial.suggest_int("grid",tuning["grid"]["min"],tuning["grid"]["max"]),trial.suggest_int("spline_order",tuning["spline_order"]["min"],tuning["spline_order"]["max"]));model=KirchhoffInversePIKAN(*architecture,mode,values,CONFIG["inverse"],device,seed).to(device)
        fn=lambda:loss(model,interior,*sensors,boundaries,values,CONFIG["loss_weights"]);training=dict(CONFIG["training"]);training["epochs"]=int(epochs or tuning["epochs"]);training["print_every"]=0;train_inverse(model,fn,training);field=float(relative_l2(model,validation,values));parameter=abs(float(model.flexural_rigidity())-values["true_d"])/values["true_d"];trial.set_user_attr("field_relative_l2",field);trial.set_user_attr("parameter_relative_error",parameter);return field+parameter
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed));study.optimize(objective,n_trials=int(trials or tuning["trials"]));payload=json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {};payload[mode]={"best_value":study.best_value,"best_params":study.best_params,"best_metrics":study.best_trial.user_attrs};RESULT_PATH.write_text(json.dumps(payload,indent=2));print(f"Saved: {RESULT_PATH}")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",choices=("cpu","gpu"));p.add_argument("--constraint",choices=("soft","hard"),default="hard");p.add_argument("--trials",type=int);p.add_argument("--epochs",type=int);a=p.parse_args();main(a.device,a.constraint,a.trials,a.epochs)
