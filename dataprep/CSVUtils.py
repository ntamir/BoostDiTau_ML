import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def trf_abs(df,varname):
    df[varname]=(abs(df[varname]))
    
def swap_feats(df, var1, var2):
    df[[var1,var2]]=df[[var2,var1]]

def scale_feat_const(df,varname,scale):
    df[varname]=df[varname]/scale

def get_class_subset(df,labelstr,labelval):
    return df[(df[labelstr]==labelval)].copy()

def create_balanced_set(df1,df2,randseed=-1):
    sample_size=min([len(df1.index),len(df2.index)])
    if randseed==-1:
        df1_red=df1.sample(n=sample_size)
        df2_red=df2.sample(n=sample_size)
        df_recomb = pd.concat([df1_red,df2_red],ignore_index=True)
        for i in range(100):
            df_recomb=df_recomb.sample(frac=1)
    else:
        df1_red=df1.sample(n=sample_size,random_state=randseed)
        df2_red=df2.sample(n=sample_size,random_state=randseed)
        df_recomb = pd.concat([df1_red,df2_red],ignore_index=True)
        for i in range(100):
            df_recomb=df_recomb.sample(frac=1,random_state=randseed)
    return df_recomb

def scale_feats(df,col_min,col_max,scaler=StandardScaler()):
    df.iloc[:,col_min:col_max] = scaler.fit_transform(df.iloc[:,1:15])
    return scaler

def get_split_datasets(df,randseed=-1,train_frac=0.7,valtest_frac=0.5):
    if randseed==-1:
        train=df.sample(frac=train_frac)
        remains=df.drop(train.index)
        val=remains.sample(frac=valtest_frac)
        test=remains.drop(val.index)
    else:
        train=df.sample(frac=train_frac,random_state=randseed)
        remains=df.drop(train.index)
        val=remains.sample(frac=valtest_frac,random_state=randseed)
        test=remains.drop(val.index)
    return train,val,test

