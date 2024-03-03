import torch
import torch.nn as nn
import numpy as np

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # uniform distribution to the weights/biases
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.uniform_(-1.0,1.0)

class DNN(nn.Module):
    def __init__(self,useSkip=False,DOfrac=0.1,HSize=32,nLayers=5):
        super().__init__()
        self.useSkip=useSkip
        self.HS=HSize
        self.nlayers=nLayers
        if self.useSkip==True:
            self.InLayer     = nn.Sequential(nn.Linear(14,32),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(32,self.HS),nn.PReLU(),nn.Dropout(DOfrac))
            self.FirstBlock  = nn.Sequential(nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac))
            self.SecondBlock = nn.Sequential(nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac))
            self.ThirdBlock  = nn.Sequential(nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac))
            self.OutBlock    = nn.Sequential(nn.Linear(self.HS,32),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(32,16),nn.PReLU(),nn.Dropout(DOfrac),
                                             nn.Linear(16,1))
        else:
            self.DNNmodel=nn.Sequential(nn.Linear(14,32),nn.PReLU(),nn.Dropout(DOfrac),
                                        nn.Linear(32,self.HS),nn.PReLU(),nn.Dropout(DOfrac))
            for i in range(self.nlayers):
                self.DNNmodel.append(nn.Sequential(nn.Linear(self.HS,self.HS),nn.PReLU(),nn.Dropout(DOfrac)))
            
            self.DNNmodel.append(nn.Sequential(nn.Linear(self.HS,32),nn.PReLU(),nn.Dropout(DOfrac),
                                               nn.Linear(32,16),nn.PReLU(),nn.Dropout(DOfrac),
                                               nn.Linear(16,1)))
        #self.DNNmodel.apply(weights_init_uniform)
    def forward(self,x):
        if self.useSkip==False:
            out = self.DNNmodel(x)
        else:
            y0 = self.InLayer(x)
            y1 = self.FirstBlock(y0)
            y2 = self.SecondBlock(y0+y1)
            y3 = self.ThirdBlock(y0+y1+y2)
            out = self.OutBlock(y3)
        return out
                                    
