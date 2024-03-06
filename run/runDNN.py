import sys
sys.path.append("../")
from runUtils.steering import *
from dataprep.plottingUtils import *
import optuna
torch.set_num_threads(torch.get_num_threads())

skip=False
version='ForOptuna_Feb2024'

def objective(trial):
    hpars = {
        "BSize":trial.suggest_int("BSize",512,1024),
        "HS":trial.suggest_int("HS",16,64),
        "lrate":trial.suggest_float("lrate",1e-3,1e-3),
        "l2weight":trial.suggest_float("l2weight",1e-5,1e-5),
        "dropout":trial.suggest_float("dropout",0.05,0.25),
        "numLayers":trial.suggest_int("numLayers",3,9)
    }
    outputs = run_training_DNN(n_epochs=100,hyperpars=hpars,skip=False)
    print("Best loss of {} achieved!".format(outputs['best_loss']))
    return outputs['best_loss']
6
if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=20)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)
    outputs = run_training_DNN(n_epochs=100,hyperpars=trial_.params,save_model=True)
    plot_history(outputs['train_prog'][0],outputs['valid_prog'][0],outputs['plot_str'],'Loss',"png")
    plot_history(outputs['train_prog'][1],outputs['valid_prog'][1],outputs['plot_str'],'Accuracy',"png")