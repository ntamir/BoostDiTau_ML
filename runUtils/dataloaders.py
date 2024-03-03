import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import dgl
from tqdm import tqdm
import gc

class MLPDataset(Dataset):
    
    def __init__(self, file_name):
        
        #read csv and load row data to variables
        infile=pd.read_csv(file_name)
        x = infile.iloc[0:len(infile),2:(len(infile.columns)-1)].values
        y = infile.iloc[0:len(infile),(len(infile.columns)-1)].values
        #make torch tensors
        self.x_tensor = torch.tensor(x, dtype=torch.float32)
        self.y_tensor = torch.tensor(y)
        
    def __len__(self):
        return len(self.y_tensor)
    
    def __getitem__(self,idx):
        return self.x_tensor[idx], self.y_tensor[idx]
    
class edgenet(nn.Module): #Quick net to calculate object-subjet deltaR's and write it as an edge feature
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        splitsrc=torch.hsplit(x.src['center'],2)
        splitdst=torch.hsplit(x.dst['center'],2)        
        deta2=torch.square(torch.add(splitsrc[0],splitdst[0],alpha=-1))
        dphi2=torch.square(torch.add(splitsrc[1],splitdst[1],alpha=-1))
        dR=torch.sqrt(torch.add(deta2,dphi2))
        x.data['dR']=dR
        return {'dR':dR}

class GNNDataset(Dataset):
    def __init__(self,path,usecpu=True,make_edges=False,thin_edges=True,minDR=0.1,rotate_graphs=False):
        
        print("Starting dataloader...")
        self.graphs, _ = dgl.data.utils.load_graphs(path)
        self.rotgraphs=[]
        print("Graphs loaded...")
        if make_edges:
            for g in tqdm(self.graphs):
                self.make_all_edges(g)
                self.calculate_dr_edges(g)
                if thin_edges:
                    self.thin_edges(g,min_dR=minDR)
        if rotate_graphs:
            print("Rotating graphs...")
            for idx, g in enumerate(tqdm(self.graphs)):
                self.graphs[idx] = rotate_graph(g, find_leadSJ_angle(g))
        gc.collect()
        if torch.cuda.is_available() and not usecpu:
            self.graphs = [g.to(torch.device('cuda')) for g in self.graphs]        
    
    def make_all_edges(self,g):
        ntypelist=g.ntypes
        for ntype_src in ntypelist:
            for ntype_dst in ntypelist:
                print("Creating edges between {src} and {dst}...".format(src=ntype_src,dst=ntype_dst))
                self.make_edges(g,src_ntype=ntype_src,dst_ntype=ntype_dst)
           
    def make_edges(self,g,usecpu=True,src_ntype="",dst_ntype=""):
        edge_src=torch.repeat_interleave(torch.arange(g.num_nodes(src_ntype)),g.num_nodes(dst_ntype))
        edge_dst=torch.arange(g.num_nodes(dst_ntype)).repeat(g.num_nodes(src_ntype))
        if torch.cuda.is_available() and not usecpu:
            edge_src=edge_src.to(torch.device('cuda'))
            edge_dst=edge_dst.to(torch.device('cuda'))
        etype="{src}2{dst}".format(src=src_ntype,dst=dst_ntype)
        g.add_edges(edge_src,edge_dst,etype=etype)
        
    def calculate_dr_edges(self,g):
        self.enet= edgenet()
        for etype in g.etypes:
            g.apply_edges(self.enet,etype=etype)
        
    def thin_all_edges(self,g,min_dR=0.2):
        for etype in g.etypes:
            feat_vals=g.edges[etype].data["dR"]
            eids_to_remove=(feat_vals>min_dR).nonzero(as_tuple=True)[0]
            g.remove_edges(eids_to_remove,etype=etype)     
            
    def __len__(self):
       
        return len(self.graphs)

    def __getitem__(self, idx):
        
        return self.graphs[idx]

class CNNDataset(Dataset):

    def __init__(self, data_pkl_str, data_pkl_list = None, transform=None):
        self.data_pkl_str = data_pkl_str
    
        if data_pkl_list == None:
            self.data_list = pd.read_pickle(self.data_pkl_str)
        else:
            self.data_list = data_pkl_list
    
    def __getitem__(self, index): 
        if isinstance(index,int):
            image = self.data_list[index]['cell_image']
            label = self.data_list[index]['truthmatch']
            
            return (image, label)

        elif isinstance(index, slice):
            new = DiTauDataset(data_pkl_str, self.data_list[index])
            
            return new
    
    def shuffle(self):
        np.random.shuffle(self.data_list)
        
    def __len__(self):
        return len(self.data_list)

    def pop(self, index):
        del(self.data_list[index])
        gc.collect()
