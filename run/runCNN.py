import sys
sys.path.append("../")
from runUtils.steering import *
import optuna
torch.set_num_threads(torch.get_num_threads())

version='ForOptuna_March2024'

def objective(trial):
    hpars = {
        "BSize":trial.suggest_int("BSize",512,1024),
        "lrate":trial.suggest_float("lrate",1e-3,1e-3),
        "l2weight":trial.suggest_float("l2weight",1e-5,1e-5),
        "dropout":trial.suggest_float("dropout",0.05,0.25),
        "filter_1_size":trial.suggest_int("filter_1_size",3,5,step=2),
        "filter_2_size":trial.suggest_int("filter_2_size",3,5,step=2),
        "filter_3_size":trial.suggest_int("filter_3_size",3,5,step=2)
    }
    outputs = run_training_CNN(n_epochs=100,hyperpars=hpars)
    print("Best loss of {} achieved!".format(best_loss))
    return outputs['best_loss']

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=20)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    outputs = run_training_CNN(n_epochs=100,hyperpars=trial_.params,save_model=True)
    plot_history(outputs['train_prog'][0],outputs['valid_prog'][0],outputs['plot_str'],'Loss',"pdf")
    plot_history(outputs['train_prog'][1],outputs['valid_prog'][1],outputs['plot_str'],'Accuracy',"pdf")
