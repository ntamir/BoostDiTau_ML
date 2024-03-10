import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_cols(df,col_list,nbin=100,norm=True,label="",alpha=0.5,grid=False):
    return df.iloc[:,col_list].hist(bins=nbin,figsize=(16,16), density=norm,label=label,alpha=alpha,grid=grid)

def plot_cols_same(df,axis,col_list,nbin=100,norm=True,label="",alpha=0.5,grid=False):
    df.iloc[:,col_list].hist(bins=nbin,figsize=(16,16), ax=axis, density=norm,label=label,alpha=alpha,grid=grid)
    
def add_axis_labels(df,ax,col_list):
    for a,c in zip(ax.flatten(),df.iloc[:,[1,3,5,7,8,12]]):
        bin_widths = [patch.get_width() for patch in a.patches]
        parts = c.split("_")
        if len(parts) == 2:
            x, y = parts
            xlabel = f"${x}_{{{y}}}$"
        elif len(parts) == 3:
            x, y, z = parts
            xlabel = f"${x}_{{{y}}}^{{{z}}}$"
        else:  
            xlabel = c 
        a.set_xlabel(xlabel,fontsize=15.0)
        a.set_ylabel(f"$\dfrac{{Events}}{{{bin_widths[0]:.3f}}}$")
        a.set_title("")
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i][j].legend()
                
def plot_history(train_history,valid_history,plotstring="./loss",ylabel='Loss',outformat="png"):
    fig,ax=plt.subplots(1,1,figsize=(24,16))
    ax.plot(train_history, label='Training',linewidth=2.5)
    ax.plot(valid_history, label='Validation',linewidth=2.5)
    ax.set_ylabel(ylabel,fontsize=24)
    ax.set_xlabel('Epoch',fontsize=24)
    ax.legend(fontsize=20)
    ax.grid()
    plt.ylim(bottom=0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plot_str="{instr}_{ylab}.{ft}".format(instr=plotstring,ylab=ylabel,ft=outformat)
    plt.savefig(plot_str) 
    
def plot_confusion_matrix(labels,predictions,norm=None,plotstring="./ConfMat",ylabel="ConfMat",outformat="png"):
    cm = confusion_matrix(labels,predictions,normalize=norm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plot_str="{instr}_{ylab}.{ft}".format(instr=plotstring,ylab=ylabel,ft=outformat)
    plt.savefig(plot_str)
    
def export_roc(labels,predictions,fstring="./ConfMat"):
    fpr, tpr, thresholds = metrics.roc_curve(labels,predictions)
    auc = metrics.auc(fpr,tpr)
    out_df=pd.DataFrame({'xpoint':tpr , 'ypoint':1/np.maximum(fpr,1e-6) })
    out_df.to_csv("{}.csv".format(fstring))
    return auc
    