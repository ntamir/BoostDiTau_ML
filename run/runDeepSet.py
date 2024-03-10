import torch
import sys
sys.path.append("../")
from runUtils.steering import *
from dataprep.plottingUtils import *
import optuna
torch.set_num_threads(torch.get_num_threads())

sampled_g = getGNNdatasets('../dataset/DGLdataset_balanced.bin',randseed=234)

def objective(trial):
    hpars = {
        "BSize":trial.suggest_int("BSize",512,2048),
        "hs_calo":trial.suggest_int("hs_calo",16,16),
        "hs_trk":trial.suggest_int("hs_trk",16,16),
        "hs_sj":trial.suggest_int("hs_sj",12,12),
        "num_modules":trial.suggest_int("num_modules",3,3),
        "module_depth":trial.suggest_int("module_depth",1,1),
        "lrate":trial.suggest_float("lrate",5e-4,5e-2,log=True),
        "l2weight":trial.suggest_float("l2weight",5e-6,5e-4,log=True),
        "dropout":trial.suggest_float("dropout",0.05,0.3,log=True),
        "GlobAtt":trial.suggest_categorical("GlobAtt",[True]),
    }
    outputs = run_training_DeepSet(sampled_g,n_epochs=100,hyperpars=hpars)
    print("Best loss of {} achieved!".format(outputs['best_loss']))
    return outputs['best_loss']

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=12)
    
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)
    outputs = run_training_DeepSet(sampled_g,n_epochs=100,hyperpars=trial_.params,save_model=True)
    plot_history(outputs['train_prog'][0],outputs['valid_prog'][0],outputs['plot_str'],'Loss',"png")
    plot_history(outputs['train_prog'][1],outputs['valid_prog'][1],outputs['plot_str'],'Accuracy',"png")
    plot_confusion_matrix(outputs['test_labels'],outputs['class_preds'],None,outputs['plot_str'],"ConfMat_noNorm",outformat="png")
    plot_confusion_matrix(outputs['test_labels'],outputs['class_preds'],'true',outputs['plot_str'],"ConfMat_Norm",outformat="png")
    auc = export_roc(outputs['test_labels'],outputs['test_preds'],outputs['ROC_str'])
    print("Study done, best AUC of {} achieved!".format(auc))