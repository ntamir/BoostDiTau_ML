
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl



class DeepSet(nn.Module):
    def __init__(self,hs_point=16,hs_trk=16,hs_sj=16,num_modules=3,module_depth=1,DOfrac=0.1,useBN=False,GlobAtt=False):
        super().__init__()
        
        hs_p = hs_point
        hs_o = hs_sj
        hs_t = hs_trk
        nd = num_modules
        md = module_depth
        self.useBN=useBN
        usebias= not useBN
        self.GlobAtt = GlobAtt
        
        self.nodeinit_p = nn.Sequential(nn.Linear(4,hs_p),nn.Dropout(p=DOfrac),nn.PReLU(),
                                        nn.Linear(hs_p,hs_p),nn.Dropout(p=DOfrac),nn.PReLU())
        self.nodeinit_t = nn.Sequential(nn.Linear(4,hs_t),nn.Dropout(p=DOfrac),nn.PReLU(),
                                        nn.Linear(hs_t,hs_t),nn.Dropout(p=DOfrac),nn.PReLU())
        self.nodeinit_o = nn.Sequential(nn.Linear(3,hs_o),nn.Dropout(p=DOfrac),nn.PReLU(),
                                        nn.Linear(hs_o,hs_o),nn.Dropout(p=DOfrac),nn.PReLU())
        
        if self.GlobAtt:
            self.gate_p = nn.Linear(hs_p,1)
            self.gate_t = nn.Linear(hs_t,1)            
            self.gate_o = nn.Linear(hs_o,1)
        
            self.GAP_p = dgl.nn.GlobalAttentionPooling(self.gate_p)            
            self.GAP_t = dgl.nn.GlobalAttentionPooling(self.gate_t)
            self.GAP_o = dgl.nn.GlobalAttentionPooling(self.gate_o)        
        
        self.hidden_layers_p = nn.ModuleList()
        self.hidden_layers_t = nn.ModuleList()
        self.hidden_layers_o = nn.ModuleList()
        if self.useBN==True:    
            self.batch_norms_p = nn.ModuleList()
            self.batch_norms_o = nn.ModuleList()
            self.batch_norms_t = nn.ModuleList()
        
        self.LRJclassifierDS = nn.Sequential(nn.Linear(hs_o+hs_p+hs_t,hs_o+hs_p+hs_t),nn.PReLU(),nn.Dropout(p=DOfrac),
                                                   #nn.Linear(hs_o+hs_p+hs_t,hs_o+hs_p+hs_t),nn.PReLU(),nn.Dropout(p=DOfrac),
                                                   nn.Linear(hs_o+hs_p+hs_t,32),nn.PReLU(),nn.Dropout(p=DOfrac),
                                                   nn.Linear(32,16),nn.PReLU(),nn.Dropout(p=DOfrac),
                                                   nn.Linear(16,1))        
        
        for i in range(nd):
            seq=nn.Sequential(nn.Linear((hs_p*4)+4,hs_p),nn.PReLU(),nn.Dropout(p=DOfrac))
            for i in range(md):
                seq.append(nn.Sequential(nn.Linear(hs_p,hs_p),nn.PReLU(),nn.Dropout(p=DOfrac)))
            seq.append(nn.Sequential(nn.Linear(hs_p,hs_p,bias=usebias),nn.Dropout(p=DOfrac),nn.PReLU()))
            self.hidden_layers_p.append(seq)
            if self.useBN==True:
                self.batch_norms_p.append(nn.BatchNorm1d(hs_p,track_running_stats=False))
            
            seq=nn.Sequential(nn.Linear((hs_t*4)+4,hs_t),nn.PReLU(),nn.Dropout(p=DOfrac))
            for i in range(md):
                seq.append(nn.Sequential(nn.Linear(hs_t,hs_t),nn.PReLU(),nn.Dropout(p=DOfrac)))
            seq.append(nn.Sequential(nn.Linear(hs_t,hs_t,bias=usebias),nn.Dropout(p=DOfrac),nn.PReLU()))
            self.hidden_layers_t.append(seq)
            if self.useBN==True:
                self.batch_norms_t.append(nn.BatchNorm1d(hs_t,track_running_stats=False))
            
            seq=nn.Sequential(nn.Linear((hs_o*4)+3,hs_o),nn.PReLU(),nn.Dropout(p=DOfrac))
            for i in range(md):
                seq.append(nn.Sequential(nn.Linear(hs_o,hs_o),nn.PReLU(),nn.Dropout(p=DOfrac)))
            seq.append(nn.Sequential(nn.Linear(hs_o,hs_o,bias=usebias),nn.Dropout(p=DOfrac),nn.PReLU()))
            self.hidden_layers_o.append(seq)
            if self.useBN==True:
                self.batch_norms_o.append(nn.BatchNorm1d(hs_o,track_running_stats=False))
            

        
       
    def forward(self,g):
        
        #Initialize inputs: Global energy normalization factor
        E_g=torch.unsqueeze(g.nodes['global'].data['E'].float(),dim=1)
        g.nodes['cells'].data['Eg']=dgl.broadcast_nodes(g,E_g,ntype='cells')
        g.nodes['tracks'].data['Eg']=dgl.broadcast_nodes(g,E_g,ntype='tracks')
        g.nodes['subjets'].data['Eg']=dgl.broadcast_nodes(g,E_g,ntype='subjets')
        g.nodes['cells'].data['TM']=dgl.broadcast_nodes(g,g.nodes['global'].data['TruthMatch'],ntype='cells')
        g.nodes['tracks'].data['TM']=dgl.broadcast_nodes(g,g.nodes['global'].data['TruthMatch'],ntype='tracks')
        g.nodes['subjets'].data['TM']=dgl.broadcast_nodes(g,g.nodes['global'].data['TruthMatch'],ntype='subjets')
        
        #Initialize point input features
        type_p=torch.unsqueeze(((g.nodes['cells'].data['type']).float())-2.5,dim=1)
        E_p=torch.unsqueeze((g.nodes['cells'].data['E']).float(),dim=1)/g.nodes['cells'].data['Eg']
        infeats_p = torch.cat([g.nodes['cells'].data['center'].float(),type_p,E_p],dim=1)
        g.nodes['cells'].data['En']=E_p.squeeze()
        #infeats_p.float()
        #print(infeats_p.dtype)
        
        #Initialize Track input features
        D0=torch.abs(torch.unsqueeze(((g.nodes['tracks'].data['D0']).float()),dim=1))
        E_t=torch.unsqueeze((g.nodes['tracks'].data['E']).float(),dim=1)/g.nodes['tracks'].data['Eg']
        infeats_t = torch.cat([g.nodes['tracks'].data['center'].float(),D0,E_t],dim=1)
        g.nodes['tracks'].data['En']=E_t.squeeze()
        
        #Initialize SubJet input features
        E_o=torch.unsqueeze((g.nodes['subjets'].data['E']).float(),dim=1)/g.nodes['subjets'].data['Eg']
        infeats_o = torch.cat([g.nodes['subjets'].data['center'].float(),E_o],dim=1)
        #infeats_o.to(torch.float32)
        #print(infeats_o)
        
        #Initialize Hidden Reps
        g.nodes['cells'].data['hidden rep'] = self.nodeinit_p(infeats_p) #Cell input features -> HRep(point)
        g.nodes['tracks'].data['hidden rep'] = self.nodeinit_t(infeats_t) #Track input features -> HRep(track)
        g.nodes['subjets'].data['hidden rep'] = self.nodeinit_o(infeats_o) #Sub-jet input features -> HRep(object)
        
        #Loop over blocks
        for i, layer in enumerate( self.hidden_layers_o):
            
            #Run the point network to update HRep(point)
            g.nodes['cells'].data['sum rep'] = dgl.broadcast_nodes(g, dgl.sum_nodes(g, 'hidden rep', ntype='cells'), ntype='cells')
            g.nodes['cells'].data['mean rep'] = dgl.broadcast_nodes(g, dgl.mean_nodes(g, 'hidden rep', ntype='cells'), ntype='cells')
            g.nodes['cells'].data['max rep'] = dgl.broadcast_nodes(g, dgl.max_nodes(g, 'hidden rep', ntype='cells'), ntype='cells')
            input_p = torch.cat([
                                        infeats_p,
                                        g.nodes['cells'].data['sum rep'],
                                        g.nodes['cells'].data['mean rep'],
                                        g.nodes['cells'].data['max rep'],
                                        g.nodes['cells'].data['hidden rep']],dim=1)
            g.nodes['cells'].data['hidden rep'] = self.hidden_layers_p[i](input_p)
            if self.useBN==True:
                print('entering BN')
                g.nodes['cells'].data['hidden rep'] = self.batch_norms_p[i](g.nodes['cells'].data['hidden rep'])
            
            #Run the track-network to update HRep(track)
            g.nodes['tracks'].data['sum rep'] = dgl.broadcast_nodes(g, dgl.sum_nodes(g, 'hidden rep', ntype='tracks'), ntype='tracks')
            g.nodes['tracks'].data['mean rep'] = dgl.broadcast_nodes(g, dgl.mean_nodes(g, 'hidden rep', ntype='tracks'), ntype='tracks')
            g.nodes['tracks'].data['max rep'] = dgl.broadcast_nodes(g, dgl.max_nodes(g, 'hidden rep', ntype='tracks'), ntype='tracks')
            input_t = torch.cat([
                                        infeats_t,
                                        g.nodes['tracks'].data['sum rep'],
                                        g.nodes['tracks'].data['mean rep'],
                                        g.nodes['tracks'].data['max rep'], 
                                        g.nodes['tracks'].data['hidden rep']],dim=1)
            g.nodes['tracks'].data['hidden rep'] = self.hidden_layers_t[i](input_t)
            if self.useBN==True:
                print('entering BN')
                g.nodes['tracks'].data['hidden rep'] = self.batch_norms_t[i](g.nodes['tracks'].data['hidden rep'])


            #Generate a GRep(point) from the updated HRep(point+trk) and propagate to object nodes, then run object-network
            #g.nodes['subjets'].data['global point rep'] = dgl.broadcast_nodes(g,dgl.sum_nodes(g, 'hidden rep', ntype='cells'),ntype='subjets')
            #g.nodes['subjets'].data['global trk rep'] = dgl.broadcast_nodes(g,dgl.sum_nodes(g, 'hidden rep', ntype='tracks'),ntype='subjets')
            g.nodes['subjets'].data['sum rep'] = dgl.broadcast_nodes(g,dgl.sum_nodes(g, 'hidden rep', ntype='subjets'),ntype='subjets')
            g.nodes['subjets'].data['mean rep'] = dgl.broadcast_nodes(g,dgl.mean_nodes(g, 'hidden rep', ntype='subjets'),ntype='subjets')
            g.nodes['subjets'].data['max rep'] = dgl.broadcast_nodes(g,dgl.max_nodes(g, 'hidden rep', ntype='subjets'),ntype='subjets')
            input_o = torch.cat([
                                        infeats_o,
                                        g.nodes['subjets'].data['sum rep'],                                  
                                        g.nodes['subjets'].data['mean rep'],                                 
                                        g.nodes['subjets'].data['max rep'],
                                        g.nodes['subjets'].data['hidden rep']],dim=1)
            g.nodes['subjets'].data['hidden rep'] = self.hidden_layers_o[i](input_o)
            if self.useBN==True:
                print('entering BN')
                g.nodes['subjets'].data['hidden rep'] = self.batch_norms_o[i](g.nodes['subjets'].data['hidden rep'])
        
        #Run the LRJ classifier on permutation-invariant aggregated node data (mean/sum/min/max etc)
        #Two global attention pooling layers (or simple sum if turned off)
        #Requires creating homogeneous graphs for each node type, and copy batch informations for each type from heterograph
        if self.GlobAtt==True:
            subg_p=dgl.graph(([], []), num_nodes=g.num_nodes(ntype='cells'))
            subg_t=dgl.graph(([], []), num_nodes=g.num_nodes(ntype='tracks'))
            subg_o=dgl.graph(([], []), num_nodes=g.num_nodes(ntype='subjets'))
            subg_p.set_batch_num_nodes(g.batch_num_nodes('cells'))
            subg_t.set_batch_num_nodes(g.batch_num_nodes('tracks'))
            subg_o.set_batch_num_nodes(g.batch_num_nodes('subjets'))
            if g.nodes['cells'].data['hidden rep'].get_device()==0:
                subg_p = subg_p.to(torch.device('cuda'))
                subg_t = subg_t.to(torch.device('cuda'))
                subg_o = subg_o.to(torch.device('cuda'))
            GATpooled_p = (self.GAP_p(subg_p,g.nodes['cells'].data['hidden rep']))
            GATpooled_t = (self.GAP_t(subg_t,g.nodes['tracks'].data['hidden rep']))
            GATpooled_o = (self.GAP_o(subg_o,g.nodes['subjets'].data['hidden rep']))
            g.nodes['global'].data['properties'] = self.LRJclassifierDS(torch.cat([GATpooled_p,GATpooled_o,GATpooled_t],dim=1))
        else:
            g.nodes['global'].data['properties'] = self.LRJclassifierDS(torch.cat([dgl.mean_nodes(g, 'hidden rep', ntype='cells'), dgl.mean_nodes(g, 'hidden rep', ntype='subjets'),dgl.mean_nodes(g, 'hidden rep', ntype='tracks')],dim=1)) 
            
        return g