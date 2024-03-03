
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc

def plot_graph(g_input,ax,size=1):
    g = g_input.to(torch.device('cpu'))
    x = g.nodes['points'].data['center'][:,0].data.numpy()
    y = g.nodes['points'].data['center'][:,1].data.numpy()
    t = g.nodes['points'].data['type'].data.numpy()
    object_centers = g.nodes['predicted objects'].data['center'].data.numpy()
    object_TruthMatch = g.nodes['predicted objects'].data['TruthMatch'].data.numpy()

    ax.scatter(x,y,c=t,cmap='Paired',s=size, vmin=2,vmax=4)
    

    for i in range(len(object_centers)):
        ec='r' if object_TruthMatch[i]==0 else 'b'
        bounding_box = patches.Circle((object_centers[i][0], object_centers[i][1]), 
                             0.2, linewidth=1, edgecolor=ec, facecolor='none')



        ax.add_patch(bounding_box)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    
def find_leadSJ_angle(g_input):
    g = g_input.to(torch.device('cpu'))
    x = g.nodes['predicted objects'].data['center'][0,0].data.numpy()
    y = g.nodes['predicted objects'].data['center'][0,1].data.numpy()
    angle = np.arctan2(y,x)
    return angle

def rotate_graph(g_input, theta):
    device = g_input.device
    g = g_input.to('cpu')  # Assume the original device is GPU
    theta = torch.tensor(theta+np.pi/2, dtype=torch.float64)
    Rmatrix = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)], 
                            [torch.sin(-theta), torch.cos(-theta)]], dtype=torch.float64)
    
    g.nodes['predicted objects'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['predicted objects'].data['center'])))
    g.nodes['points'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['points'].data['center'])))
    g.nodes['tracks'].data['center'] = torch.t(torch.mm(Rmatrix, torch.t(g.nodes['tracks'].data['center'])))
    
    g_input = g.to(device)  # Convert back to original device
    return g_input

def collate_graphs(samples):    
    batched_graph = dgl.batch(samples)    
    return batched_graph
 
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

class DiTauDataset(Dataset):
    def __init__(self,path,usecpu=True,trf_trk=False,make_edges=False,thin_edges=True,minDR=0.1,rotate_graphs=False):
        
        print("Starting dataloader...")
        self.graphs, _ = dgl.data.utils.load_graphs(path)
        self.oldgraphs=[]
        self.newgraphs=[]
        self.rotgraphs=[]
        print("Graphs loaded...")
        if trf_trk:
            for g in tqdm(self.graphs):
                self.newgraphs.append(self.transform_tracks(g))
            self.oldgraphs=self.graphs
            self.graphs=self.newgraphs
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
    
    def transform_tracks(self,g,usecpu=True):
        node_feats = g.nodes['points'].data
        type_vals = node_feats['type'].squeeze()
        nodes_to_remove = (type_vals==4).nonzero(as_tuple=True)[0]
        
        #Add also new edge types- trk2point/point2trk/trk2trk/trk2obj
        canonical_etypes = list(g.canonical_etypes)
        canonical_etypes.append(('tracks','tracks_to_tracks','tracks'))
        canonical_etypes.append(('points','points_to_tracks','tracks'))
        canonical_etypes.append(('tracks','tracks_to_points','points'))
        canonical_etypes.append(('tracks','tracks_to_object','predicted objects'))
        
        num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
        num_nodes_dict['tracks']=0
        edges_dict={etype:([],[]) for etype in canonical_etypes}
        g_new=dgl.heterograph(edges_dict,num_nodes_dict=num_nodes_dict)
        #print(g_new.canonical_etypes)
        
        for ntype in g.ntypes:
            for k,v in g.nodes[ntype].data.items():
                g_new.nodes[ntype].data[k] = v
        
        new_node_feats = {k: v[nodes_to_remove] for k,v in node_feats.items()} 
        g_new = dgl.add_nodes(g_new, len(nodes_to_remove), data=new_node_feats, ntype='tracks')
        g_new = dgl.remove_nodes(g_new,nodes_to_remove,ntype='points')
        g = g_new
        
        #print(g)
        return g_new
    
            
    def make_all_edges(self,g,usecpu=True):
        p2p_src=torch.repeat_interleave(torch.arange(g.num_nodes("points")),g.num_nodes("points"))
        p2p_dst=torch.arange(g.num_nodes("points")).repeat(g.num_nodes("points"))
        p2o_src=torch.repeat_interleave(torch.arange(g.num_nodes("points")),g.num_nodes("predicted objects"))
        p2o_dst=torch.arange(g.num_nodes("predicted objects")).repeat(g.num_nodes("points"))
        o2g_src=torch.repeat_interleave(torch.arange(g.num_nodes("predicted objects")),g.num_nodes("global"))
        o2g_dst=torch.arange(g.num_nodes("global")).repeat(g.num_nodes("predicted objects"))
        if torch.cuda.is_available() and not usecpu:
            p2p_src=p2p_src.to(torch.device('cuda'))
            p2p_dst=p2p_dst.to(torch.device('cuda'))
            p2o_src=p2o_src.to(torch.device('cuda'))
            p2o_dst=p2o_dst.to(torch.device('cuda'))
            o2g_src=o2g_src.to(torch.device('cuda'))
            o2g_dst=o2g_dst.to(torch.device('cuda'))
        g.add_edges(p2p_src,p2p_dst,etype="points_to_points")
        g.add_edges(p2o_src,p2o_dst,etype="points_to_object")
        g.add_edges(o2g_src,o2g_dst,etype="object_to_global")
        
    def calculate_dr_edges(self,g):
        self.enet= edgenet()
        g.apply_edges(self.enet,etype="points_to_object")
        g.apply_edges(self.enet,etype="points_to_points")
        g.apply_edges(self.enet,etype="object_to_global")
        
    def thin_edges(self,g,min_dR=0.2,edge_type="points_to_points"):
        feat_vals=g.edges["points_to_points"].data["dR"]
        eids_to_remove=(feat_vals>min_dR).nonzero(as_tuple=True)[0]
        g.remove_edges(eids_to_remove,etype=edge_type)
    
    def get_scale_params(self,scaler_p,scaler_o):
        eta_p=np.empty(0)
        phi_p=np.empty(0)
        energy_p=np.empty(0)
        sigtype=np.empty(0)
        eta_o=np.empty(0)
        phi_o=np.empty(0)
        energy_o=np.empty(0)        
        for g in tqdm(self.graphs):
            eta_p=np.concatenate((eta_p,g.nodes['points'].data['center'].cpu().detach().numpy()[:,0]))
            phi_p=np.concatenate((phi_p,g.nodes['points'].data['center'].cpu().detach().numpy()[:,1]))
            energy_p=np.concatenate((energy_p,g.nodes['points'].data['E'].cpu().detach().numpy())/g.nodes['global'].data['E'].cpu().detach().numpy())
            sigtype_p=np.concatenate((sigtype,g.nodes['points'].data['type'].cpu().detach().numpy()))
            eta_o=np.concatenate((eta_o,g.nodes['predicted objects'].data['center'].cpu().detach().numpy()[:,0]))
            phi_o=np.concatenate((phi_o,g.nodes['predicted objects'].data['center'].cpu().detach().numpy()[:,1]))
            energy_o=np.concatenate((energy_o,g.nodes['predicted objects'].data['E'].cpu().detach().numpy())/g.nodes['global'].data['E'].cpu().detach().numpy())
        #print(np.column_stack((eta,phi,energy,sigtype)))
        scaler_p.fit(np.column_stack((eta_p,phi_p,energy_p,sigtype)))
        scaler_o.fit(np.column_stack((eta_o,phi_o,energy_o)))
        return scaler_p.mean_ , scaler_p.scale_        
        
    def scale_features(self,g,scaler_p,scaler_o):
        E_g=torch.unsqueeze(g.nodes['global'].data['E'].float(),dim=1)
        g.nodes['points'].data['Eg']=dgl.broadcast_nodes(g,E_g,ntype='points')
        g.nodes['predicted objects'].data['Eg']=dgl.broadcast_nodes(g,E_g,ntype='predicted objects')
        
        #g.nodes['points'].data['center'][:,0]= (g.nodes['points'].data['center'][:,0]-scaler_p.mean_[0])/scaler_p.scale_[0]
        #g.nodes['points'].data['center'][:,1]= (g.nodes['points'].data['center'][:,1]-scaler_p.mean_[1])/scaler_p.scale_[1]
        g.nodes['points'].data['E']= (g.nodes['points'].data['E']-scaler_p.mean_[2])/scaler_p.scale_[2]
        #g.nodes['points'].data['E']= g.nodes['points'].data['E']/g.nodes['points'].data['Eg']
        g.nodes['points'].data['type']= (g.nodes['points'].data['type'][:,0]-scaler_p.mean_[3])/scaler_p.scale_[3]
        
        #g.nodes['predicted objects'].data['center'][:,0]= (g.nodes['predicted objects'].data['center'][:,0]-scaler_o.mean_[0])/scaler_o.scale_[0]
        #g.nodes['predicted objects'].data['center'][:,1]= (g.nodes['predicted objects'].data['center'][:,1]-scaler_o.mean_[1])/scaler_o.scale_[1]
        g.nodes['predicted objects'].data['E']= (g.nodes['predicted objects'].data['E']-scaler_o.mean_[2])/scaler_o.scale_[2]
        #g.nodes['predicted objects'].data['E']= g.nodes['predicted objects'].data['E']/g.nodes['predicted objects'].data['Eg']
    def __len__(self):
       
        return len(self.graphs)


    def __getitem__(self, idx):
        
 

        
        return self.graphs[idx]