import sys
sys.path.append("../")
from runUtils.steering import *
import optuna
torch.set_num_threads(torch.get_num_threads())

skip=False
version='ForOptuna_March2024'

def objective(trial):
    hpars = {
        "BSize":32,
        "HS":48,
        "lrate":1e-3,
        "l2weight":1e-5,
        "dropout":0.1,
        "filter_1_size":trial.suggest_int("filter_1_size",3,5,step=2),
        "filter_2_size":trial.suggest_int("filter_2_size",3,5,step=2),
        "filter_3_size":trial.suggest_int("filter_3_size",3,5,step=2)
    }
    best_loss = run_training_CNN(n_epochs=100,hyperpars=hpars,skip=skip)
    print("Best loss of {} achieved!".format(best_loss))
    return best_loss

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=1)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)
