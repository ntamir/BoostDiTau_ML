from tqdm import tqdm
#from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import random
import torch
import dgl
from dgl.data.utils import *



def plot_calo_constits(g_input,ax,size=1):
    g = g_input.to(torch.device('cpu')) # Assume the original device is GPU
    x = g.nodes['cells'].data['center'][:,0].data.numpy()
    y = g.nodes['cells'].data['center'][:,1].data.numpy()
    t = g.nodes['cells'].data['type'].data.numpy()
    object_centers = g.nodes['subjets'].data['center'].data.numpy()
    object_TruthMatch = g.nodes['subjets'].data['TruthMatch'].data.numpy()

    ax.scatter(x,y,c=t,cmap='Paired',s=size, vmin=2,vmax=4)
    

    for i in range(len(object_centers)):
        ec='r' if object_TruthMatch[i]==0 else 'b'
        bounding_box = patches.Circle((object_centers[i][0], object_centers[i][1]), 
                             0.2, linewidth=1, edgecolor=ec, facecolor='none')
        ax.add_patch(bounding_box)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    
def find_leadSJ_angle(g_input):
    g = g_input.to(torch.device('cpu')) # Assume the original device is GPU
    x = g.nodes['subjets'].data['center'][0,0].data.numpy()
    y = g.nodes['subjets'].data['center'][0,1].data.numpy()
    angle = np.arctan2(y,x)
    return angle

def rotate_graph(g_input, theta):
    device = g_input.device
    g = g_input.to('cpu')  # Assume the original device is GPU
    theta = torch.tensor(theta+np.pi/2, dtype=torch.float64)
    Rmatrix = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)], 
                            [torch.sin(-theta), torch.cos(-theta)]], dtype=torch.float64)
    
    g.nodes['subjets'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['subjets'].data['center'])))
    g.nodes['cells'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['cells'].data['center'])))
    g.nodes['tracks'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['tracks'].data['center'])))
    
    g_input = g.to(device)  # Convert back to original device
    return g_input

def collate_graphs(samples):    
    batched_graph = dgl.batch(samples)    
    return batched_graph

def merge_datasets(fCSV1="",fCSV2="",isGlobal=False):
    f1=pd.read_csv(fCSV1)
    f2=pd.read_csv(fCSV2)
    max_graph_id_f1 = f1['graph_id'].max()  # get the maximum value of 'graph_id' from f1
    f2['graph_id'] = f2['graph_id'] + max_graph_id_f1 + 1
    combined_csv = pd.concat([f1,f2],ignore_index=True)
    if isGlobal==False:
        p=combined_csv.sort_values(by=['graph_id','node_id'],ignore_index=True)
    else:
        p=combined_csv.sort_values(by=['graph_id'],ignore_index=True)
    return p

def get_nNodes(dask_df):
    n_nodes=(dask_df.groupby('graph_id').size().compute()).values.tolist()
    return n_nodes
 
def create_node_featdict(dask_df,n_dict,ntype='cells'):
    node_featdict={}
    for feat in n_dict[ntype]:
        if feat=='node_id':
            continue
        else:
            x=dask_df.groupby('graph_id').agg({feat: list}).compute()
            node_featdict[feat]=x
    return node_featdict

def create_graph(i,n_featdict={},n_sizedict={},e_dict={('cells', 'cells2cells', 'cells'): ([], [])}): #the two n_dicts must have identical keys    
    #Initialize empty graph first
    num_nodes_dict={}
    for key in n_featdict.keys():
        num_nodes_dict[key] = 0
    g = dgl.heterograph(e_dict, num_nodes_dict=num_nodes_dict)
    
    #Then fill it up with values
    curdict={}    
    for key in n_featdict.keys():
        for feat in n_featdict[key]:
            if feat=='center':
                x=torch.from_numpy(np.array([list(map(float, s.split(','))) for s in n_featdict[key][feat].values[i][0]])) #first index is graph ID
            else:
                x=torch.from_numpy(np.concatenate(n_featdict[key][feat].values[i]).T)
            curdict[feat]=x
        g.add_nodes(n_sizedict[key][i],curdict,ntype=key)
        curdict.clear()

    return g

def get_TMBalanced_dataset(glist,node_name='subjets',tm_key='TruthMatch',bkg_key='global',randseed=-1):
    tightset=[]
    bkgset=[]
    for i in tqdm(range(len(glist))):
        g=glist[i]
        isTMtight=((g.nodes[node_name].data[tm_key])[0]==1 and (g.nodes[node_name].data[tm_key])[1]==1) if g.nodes[bkg_key].data[tm_key]==1 else False
        isBKG=(g.nodes[bkg_key].data[tm_key]==0)
        if isTMtight==True:
            tightset.append(g)
        elif isBKG==True:
            bkgset.append(g)
    sample_size=min([len(tightset),len(bkgset)])
    if randseed==-1:
        tightset_red=random.sample(tightset,sample_size)
        bkgset_red=random.sample(bkgset,sample_size)
    else:
        random.seed(randseed)
        tightset_red=random.sample(tightset,sample_size)
        bkgset_red=random.sample(bkgset,sample_size)                
    tightset_balanced = tightset_red+bkgset_red
    return tightset_balanced