import torch
import torch.nn as nn
import torch.optim as optim
from models.DNN import DNN
from runUtils.dataloaders import *

def run_training_DNN(n_epochs,hyperpars,skip=False):
    train_ds = MLPDataset("../dataset_train_balanced_v02.csv")
    valid_ds = MLPDataset("../dataset_valid_balanced_v02.csv")
    test_ds = MLPDataset("../dataset_test_balanced_v02.csv")
    train_dataloader = DataLoader(train_ds,batch_size=hyperpars['BSize'],shuffle=True,pin_memory=True,num_workers=16)
    valid_dataloader = DataLoader(valid_ds,batch_size=hyperpars['BSize'])
    test_dataloader = DataLoader(test_ds,batch_size=hyperpars['BSize'])
    
    net = DNN(useSkip=skip,DOfrac=hyperpars['dropout'],HSize=hyperpars['HS'],nLayers=hyperpars['numLayers'])
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Initialized DNN with {} trainable parameters!".format(trainable_params))
    
    optimizer = optim.Adam(net.parameters(), lr=hyperpars['lrate'], weight_decay=hyperpars['l2weight'])
    loss_f = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        net.cuda()
        
    steerer = DNNSteerer(model=net,optimizer=optimizer,loss_fn=loss_f)
    pbar=tqdm(range(n_epochs))
    
    training_loss_vs_epoch = []
    validation_loss_vs_epoch = []
    training_acc_vs_epoch = []
    validation_acc_vs_epoch = []
    best_loss=np.inf
    
    for epoch in pbar:
        preloss = steerer.train(train_dataloader)
        
        train_acc, train_loss = steerer.compute_accuracy_and_loss(train_dataloader)
        valid_acc, valid_loss = steerer.compute_accuracy_and_loss(valid_dataloader)
        
        training_loss_vs_epoch.append( train_loss)    
        training_acc_vs_epoch.append( train_acc ) 
        validation_acc_vs_epoch.append(valid_acc)    
        validation_loss_vs_epoch.append(valid_loss)
        
        if len(validation_loss_vs_epoch) > 0:
            print(epoch, 'train loss',training_loss_vs_epoch[-1],'validation loss',validation_loss_vs_epoch[-1])
            print(epoch, 'train acc',training_acc_vs_epoch[-1],'validation acc',validation_acc_vs_epoch[-1])
        
        if len(validation_loss_vs_epoch)==1 or np.amin(validation_loss_vs_epoch[:-1]) > validation_loss_vs_epoch[-1]:
            best_loss=valid_loss
    
    return best_loss
    

class DNNSteerer:
    def __init__(self,model,optimizer,loss_fn):
        self.model = model
        self.optimizer=optimizer
        self.loss_fn=loss_fn
    
    def compute_accuracy_and_loss(self,dataloader,threshold=0.5):
        total = 0
        correct = 0
        loss = 0
        net=self.model
        
        sig=nn.Sigmoid()
        if torch.cuda.is_available():
            net.cuda()
            sig.to(torch.device('cuda'))
        
        net.eval()
        n_batches=0
        with torch.no_grad():
            for x,y in dataloader:
                n_batches+=1      
                if torch.cuda.is_available():
                    x=x.cuda()
                    y=y.cuda()
                    self.loss_fn.to(torch.device('cuda'))
                logits = net(x).view(-1)
                loss += self.loss_fn(logits,y.float()).item()
                pred=sig(logits)
                pred=torch.where(pred>threshold,1,0)
                correct+=len(torch.where(pred==y)[0])
                total+=len(y)
        loss=loss/n_batches
        accuracy = correct/total
        return accuracy, loss
    
    def train(self,dataloader):
        self.model.train()
        n_batches=0
        final_loss=0
        for x,y in dataloader:
            n_batches+=1
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            self.optimizer.zero_grad()
            logits = self.model(x).view(-1)
            loss = self.loss_fn(logits,y.float())
            loss.backward()
            self.optimizer.step()
            final_loss+=loss.item()
        return final_loss / n_batches
        