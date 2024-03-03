import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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