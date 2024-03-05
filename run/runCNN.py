import sys
sys.path.append("../")
from runUtils.steering import *
import optuna

# "BSize":1024,
# "HS":48,
# "lrate":1e-3,
# "l2weight":1e-5,
# "dropout":0.1,
# "numLayers":5,
skip=False
version='ForOptuna_March2024'

def objective(trial):
    hpars = {
        "BSize":32,
        "HS":48,
        "lrate":1e-3,
        "l2weight":1e-5,
        "dropout":0.1,
        "numLayers":trial.suggest_int("numLayers",4,5)
    }
    best_loss = run_training_CNN(n_epochs=100,hyperpars=hpars,skip=skip)
    print("Best loss of {} achieved!".format(best_loss))
    return best_loss

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=2)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)
