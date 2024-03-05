import sys
sys.path.append("../")
from runUtils.steering import *
import optuna
torch.set_num_threads(torch.get_num_threads())

skip=False
version='ForOptuna_Feb2024'

def objective(trial):
    hpars = {
        "BSize":1024,
        "HS":trial.suggest_int("HS",16,64),
        "lrate":1e-3,
        "l2weight":1e-5,
        "dropout":0.1,
        "numLayers":trial.suggest_int("numLayers",3,6)
    }
    best_loss = run_training_DNN(n_epochs=2,hyperpars=hpars,skip=False)
    print("Best loss of {} achieved!".format(best_loss))
    return best_loss

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=1)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)