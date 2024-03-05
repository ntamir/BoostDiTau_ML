import torch
import sys
sys.path.append("../")
from runUtils.steering import *
import optuna
torch.set_num_threads(torch.get_num_threads())

sampled_g = getGNNdatasets('../dataset/DGLdataset_balanced.bin',randseed=234)

def objective(trial):
    hpars = {
        "BSize":trial.suggest_int("BSize",1024,1024),
        "hs_calo":trial.suggest_int("hs_calo",16,64),
        "hs_trk":trial.suggest_int("hs_trk",16,64),
        "hs_sj":trial.suggest_int("hs_sj",12,48),
        "num_modules":trial.suggest_int("num_modules",2,5),
        "module_depth":trial.suggest_int("module_depth",1,4),
        "lrate":trial.suggest_float("lrate",1e-3,1e-3),
        "l2weight":trial.suggest_float("l2weight",1e-5,1e-5),
        "dropout":trial.suggest_float("dropout",0.1,0.1),
        "GlobAtt":trial.suggest_categorical("GlobAtt",[True]),
    }
    best_loss = run_training_DeepSet(sampled_g,n_epochs=100,hyperpars=hpars)
    print("Best loss of {} achieved!".format(best_loss))
    return best_loss

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=20)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)
    best_loss = run_training_DeepSet(sampled_g,n_epochs=100,hyperpars=trial_.params,save_model=True)