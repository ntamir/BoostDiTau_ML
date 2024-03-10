import torch
import torch.nn as nn
import torch.optim as optim
import random
import math as m
from datetime import datetime
from models.DNN import DNN
from models.CNN import CNN
from models.DeepSet import DeepSet
from runUtils.dataloaders import *

def getGNNdatasets(path_to_binfile,randseed=-1):
    dataset = GNNDataset(path_to_binfile,usecpu=True,make_edges=False,thin_edges=True,minDR=0.1,rotate_graphs=False)
    if randseed==-1:
        sampled_g=random.sample(dataset[:],len(dataset))
        random.shuffle(sampled_g)
        for i in tqdm(range(100)):
            random.shuffle(sampled_g)
    else:
        random.seed(randseed)
        sampled_g=random.sample(dataset[:],len(dataset))
        random.shuffle(sampled_g)
        for i in tqdm(range(100)):
            random.shuffle(sampled_g)

    return sampled_g


def run_training_DeepSet(sampled_g,n_epochs,hyperpars,save_model=False):
    
    train_dataloader = DataLoader(sampled_g[0:m.floor(0.7*len(sampled_g))], batch_size=hyperpars['BSize'], shuffle=True,
                         collate_fn=collate_graphs,pin_memory=True,num_workers=24)
    valid_dataloader = DataLoader(sampled_g[m.ceil(0.7*len(sampled_g)):m.floor(0.85*len(sampled_g))], batch_size=hyperpars['BSize'], shuffle=False, collate_fn=collate_graphs, pin_memory=False)
    test_dataloader = DataLoader(sampled_g[m.ceil(0.85*len(sampled_g)):len(sampled_g)], batch_size=hyperpars['BSize'], shuffle=False, collate_fn=collate_graphs,pin_memory=False)

    dtstr = (datetime.now()).strftime("%d%m%Y_%H%M%S")
    freetxt='ForOptuna_Feb2024'
    hyperpar_str='DeepSet3M_{GA}_HS{hsp}x{hst}x{hso}_NDS{nds}_nLayer{nla}_DO{DO}_weight{regw}_Adam_lr{lr}_BS{bs}_{dt}_{ft}'.format(GA=('_GlobAtt' if hyperpars['GlobAtt'] else ''),hsp=hyperpars['hs_calo'],hst=hyperpars['hs_trk'],hso=hyperpars['hs_sj'],nds=hyperpars['num_modules'],nla=hyperpars['module_depth'],DO=hyperpars['dropout'],regw=hyperpars['l2weight'],lr=hyperpars['lrate'],bs=hyperpars['BSize'],dt=dtstr,ft=freetxt)
    loss_str='../trained_models/loss_{}.pt'.format(hyperpar_str)
    ROC_str='../trained_models/ROC_{}'.format(hyperpar_str)
    plot_str='../trained_models/{}'.format(hyperpar_str)
    
    net = DeepSet(hs_point=hyperpars['hs_calo'],hs_trk=hyperpars['hs_trk'],hs_sj=hyperpars['hs_sj'],num_modules=hyperpars['num_modules'],module_depth=hyperpars['module_depth'],DOfrac=hyperpars['dropout'],GlobAtt=hyperpars['GlobAtt'])
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Initialized DeepSet with {} trainable parameters!".format(trainable_params))
    
    optimizer = optim.Adam(net.parameters(), lr=hyperpars['lrate'], weight_decay=hyperpars['l2weight'])
    loss_f = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        net.cuda()
        loss_f.to(torch.device('cuda'))
        
    steerer = DeepSetSteerer(model=net,optimizer=optimizer,loss_fn=loss_f)
    pbar=tqdm(range(n_epochs))
    
    training_loss_vs_epoch = []
    validation_loss_vs_epoch = []
    training_acc_vs_epoch = []
    validation_acc_vs_epoch = []
    best_loss=np.inf
    
    init_t_acc, init_t_loss = steerer.compute_accuracy_and_loss(train_dataloader)
    init_v_acc, init_v_loss = steerer.compute_accuracy_and_loss(train_dataloader)
    training_loss_vs_epoch.append( init_t_loss)    
    training_acc_vs_epoch.append( init_t_acc ) 
    validation_loss_vs_epoch.append( init_v_loss )
    validation_acc_vs_epoch.append( init_v_acc )
    
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
            if save_model:
                torch.save(net.state_dict(), loss_str)
    if save_model:
        net.load_state_dict(torch.load(loss_str,map_location='cpu'))
    acc, loss, preds, cpreds, labels = steerer.compute_test_metrics(test_dataloader)
    if save_model:
        print("Test-set loss for saved model: {}".format(loss))
        print("Test-set accuracy for saved model: {}".format(acc))
    return {'test_labels':labels,'test_preds':preds,'class_preds':cpreds,'best_loss':best_loss,'train_prog':[training_loss_vs_epoch,training_acc_vs_epoch],'valid_prog':[validation_loss_vs_epoch,validation_acc_vs_epoch],"loss_str":loss_str,"ROC_str":ROC_str,"plot_str":plot_str}

def run_training_DNN(n_epochs,hyperpars,skip=False,save_model=False):
    train_ds = MLPDataset("../dataset_train_balanced_v02.csv")
    valid_ds = MLPDataset("../dataset_valid_balanced_v02.csv")
    test_ds = MLPDataset("../dataset_test_balanced_v02.csv")
    train_dataloader = DataLoader(train_ds,batch_size=hyperpars['BSize'],shuffle=True,pin_memory=True,num_workers=16)
    valid_dataloader = DataLoader(valid_ds,batch_size=hyperpars['BSize'])
    test_dataloader = DataLoader(test_ds,batch_size=hyperpars['BSize'])
    
    dtstr = (datetime.now()).strftime("%d%m%Y_%H%M%S")
    freetxt='ForOptuna_Feb2024'
    hyperpar_str='DNN_HS{hs}xDepth{nl}{sk}DO{DO}_weight{regw}_scFeats_noposw_Adam_lr{lr}_BS{bs}_{dt}_{ft}'.format(hs=hyperpars['HS'],nl=hyperpars['numLayers'],sk=('_skip_' if skip else '_'),DO=hyperpars['dropout'],regw=hyperpars['l2weight'],lr=hyperpars['lrate'],bs=hyperpars['BSize'],dt=dtstr,ft=freetxt)
    loss_str='../trained_models/loss_{}.pt'.format(hyperpar_str)
    ROC_str='../trained_models/ROC_{}'.format(hyperpar_str)
    plot_str='../trained_models/{}'.format(hyperpar_str)
    
    net = DNN(useSkip=skip,DOfrac=hyperpars['dropout'],HSize=hyperpars['HS'],nLayers=hyperpars['numLayers'])
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Initialized DNN with {} trainable parameters!".format(trainable_params))
    
    optimizer = optim.Adam(net.parameters(), lr=hyperpars['lrate'], weight_decay=hyperpars['l2weight'])
    loss_f = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        net.cuda()
        loss_f.to(torch.device('cuda'))
        
    steerer = DNNSteerer(model=net,optimizer=optimizer,loss_fn=loss_f)
    pbar=tqdm(range(n_epochs))
    
    training_loss_vs_epoch = []
    validation_loss_vs_epoch = []
    training_acc_vs_epoch = []
    validation_acc_vs_epoch = []
    best_loss=np.inf
    
    init_t_acc, init_t_loss = steerer.compute_accuracy_and_loss(train_dataloader)
    init_v_acc, init_v_loss = steerer.compute_accuracy_and_loss(train_dataloader)
    training_loss_vs_epoch.append( init_t_loss)    
    training_acc_vs_epoch.append( init_t_acc ) 
    validation_loss_vs_epoch.append( init_v_loss )
    validation_acc_vs_epoch.append( init_v_acc )       
    
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
            if save_model:
                torch.save(net.state_dict(), loss_str)
    
    if save_model:
        net.load_state_dict(torch.load(loss_str,map_location='cpu'))
    acc, loss,preds,cpreds,labels = steerer.compute_test_metrics(test_dataloader)
    if save_model:
        print("Test-set loss for saved model: {}".format(loss))
        print("Test-set accuracy for saved model: {}".format(acc))
    return {'test_labels':labels,'test_preds':preds,'class_preds':cpreds,'best_loss':best_loss,'train_prog':[training_loss_vs_epoch,training_acc_vs_epoch],'valid_prog':[validation_loss_vs_epoch,validation_acc_vs_epoch],"loss_str":loss_str,"ROC_str":ROC_str,"plot_str":plot_str}


def run_training_CNN(n_epochs,hyperpars,save_model=False):
    data_pkl_str = '../all_samples_renormalized.pkl'

    dataset=CNNDataset(data_pkl_str)
    for i in tqdm(range(100)):
        dataset.shuffle()

    train_dataloader = DataLoader(dataset[:int(0.7*len(dataset))],batch_size=hyperpars['BSize'],shuffle=True,pin_memory=True,num_workers=16)
    valid_dataloader = DataLoader(dataset[int(0.7*len(dataset)):int(0.9*len(dataset))],batch_size=hyperpars['BSize'])
    test_dataloader = DataLoader(dataset[int(0.9*len(dataset)):],batch_size=hyperpars['BSize'])

    dtstr = (datetime.now()).strftime("%d%m%Y_%H%M%S")
    freetxt='ForOptuna_Mar2024'
    hyperpar_str='DO{DO}_weight{regw}_lr{lr}_BS{bs}_filt1-{filt1}_filt2-{filt2}_filt3-{filt3}_{dt}_{ft}'.format(DO=hyperpars['dropout'],regw=hyperpars['l2weight'],lr=hyperpars['lrate'],bs=hyperpars['BSize'],dt=dtstr,ft=freetxt,filt1=hyperpar['filter_1_size'],filt2=hyperpar['filter_2_size'],filt3=hyperpar['filter_3_size'])
    loss_str='../trained_models/loss_{}.pt'.format(hyperpar_str)
    ROC_str='../trained_models/ROC_{}.csv'.format(hyperpar_str)
    plot_str='../trained_models/{}'.format(hyperpar_str)
    
    net = CNN(in_n_pix=64, in_n_channels=3, filter_1_size=hyperpar['filter_1_size'], filter_2_size=hyperpar['filter_2_size'], filter_3_size=hyperpar['filter_3_size'],DOfrac=hyperpar['dropout'])
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Initialized CNN with {} trainable parameters!".format(trainable_params))
    
    optimizer = optim.Adam(net.parameters(), lr=hyperpars['lrate'], weight_decay=hyperpars['l2weight'])
    loss_f = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        net.cuda()
        loss_f.to(torch.device('cuda'))
        
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
            if save_model:
                torch.save(net.state_dict(), loss_str)
    
    return {'best_loss':best_loss,'train_prog':[training_loss_vs_epoch,training_acc_vs_epoch],'valid_prog':[validation_loss_vs_epoch,validation_acc_vs_epoch],"loss_str":loss_str,"ROC_str":ROC_str,"plot_str":plot_str}

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
    
    def compute_test_metrics(self,dataloader,threshold=0.5):
        total = 0
        correct = 0
        loss = 0
        net=self.model
        preds = []
        class_preds=[]
        labels=[]
        
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
                labels.extend(y.flatten().tolist())
                loss += self.loss_fn(logits,y.float()).item()
                pred=sig(logits)
                preds.extend(pred.flatten().tolist())
                pred=torch.where(pred>threshold,1,0)
                class_preds.extend(pred.flatten().tolist())
                correct+=len(torch.where(pred==y)[0])
                total+=len(y)
        loss=loss/n_batches
        accuracy = correct/total
        return accuracy, loss, preds, class_preds, labels
    
    
class DeepSetSteerer:
    def __init__(self,model,optimizer,loss_fn):
        self.model = model
        self.optimizer=optimizer
        self.loss_fn=loss_fn
    
    def compute_accuracy_and_loss(self,dataloader,threshold=0.5):
        loss = 0
        accuracy = 0
        n_batches = 0
        total = 0
        correct = 0
        net=self.model
        
        sig=nn.Sigmoid()
        if torch.cuda.is_available():
            net.cuda()
            sig.to(torch.device('cuda'))
        
        net.eval()
        n_batches=0
        with torch.no_grad():
            for batched_g in dataloader:
                n_batches+=1      
                if torch.cuda.is_available():
                    batched_g = batched_g.to(torch.device('cuda'))
                    self.loss_fn.to(torch.device('cuda'))
                predicted_g = net(batched_g)
                logits = predicted_g.nodes['global'].data['properties'].view(-1)
                labels = predicted_g.nodes['global'].data['TruthMatch']
                loss += self.loss_fn(logits,labels.float()).item()
                pred=sig(logits)
                pred=torch.where(pred>threshold,1,0)
                correct+=len(torch.where(pred==labels)[0])
                total+=len(labels)
        loss=loss/n_batches
        accuracy = correct/total
        return accuracy, loss
    
    def train(self,dataloader):
        self.model.train()
        n_batches=0
        final_loss=0
        for batched_g in dataloader:
            n_batches+=1
            if torch.cuda.is_available():
                batched_g = batched_g.to(torch.device('cuda'))
            self.optimizer.zero_grad()
            predicted_g = self.model(batched_g)
            logits = predicted_g.nodes['global'].data['properties'].view(-1)
            labels = predicted_g.nodes['global'].data['TruthMatch']
            loss = self.loss_fn(logits,labels.float())
            loss.backward()
            self.optimizer.step()
            final_loss+=loss.item()
        return final_loss / n_batches
    
    def compute_test_metrics(self,dataloader,threshold=0.5):
        loss = 0
        accuracy = 0
        n_batches = 0
        total = 0
        correct = 0
        net=self.model
        preds=[]
        class_preds=[]
        labels_out=[]
        
        sig=nn.Sigmoid()
        if torch.cuda.is_available():
            net.cuda()
            sig.to(torch.device('cuda'))
        
        net.eval()
        n_batches=0
        with torch.no_grad():
            for batched_g in dataloader:
                n_batches+=1      
                if torch.cuda.is_available():
                    batched_g = batched_g.to(torch.device('cuda'))
                    self.loss_fn.to(torch.device('cuda'))
                predicted_g = net(batched_g)
                logits = predicted_g.nodes['global'].data['properties'].view(-1)
                labels = predicted_g.nodes['global'].data['TruthMatch']
                labels_out.extend(labels.flatten().tolist())
                loss += self.loss_fn(logits,labels.float()).item()
                pred=sig(logits)
                preds.extend(pred.flatten().tolist())
                pred=torch.where(pred>threshold,1,0)
                class_preds.extend(pred.flatten().tolist())
                correct+=len(torch.where(pred==labels)[0])
                total+=len(labels)
        loss=loss/n_batches
        accuracy = correct/total
        return accuracy, loss, preds, class_preds, labels_out
