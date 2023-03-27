# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import os
import time
import torch
import numpy as np
import scipy.io as io
import torch.nn.functional as F


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.init import xavier_normal_
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import Conv1d
from early_stopping import EarlyStopping


def pixelbinning(decay,k):
    f = torch.ones([1,1,k]).type_as(decay) 
    decay_binning = F.conv1d(decay, f, stride=[k])
    return decay_binning.squeeze(0)


#-----------------------------------------------------------------------------#
def load_data(path, data_name,test_ratio = 0.2,BATCH_SIZE = 128):
    DataSet = io.loadmat(os.path.join(path,data_name))
    decay=DataSet.get('y')
    decay=decay.reshape(decay.shape[0],1,decay.shape[1])
    
    irf=DataSet.get('I')
    irf=irf.reshape(irf.shape[0],1,irf.shape[1])
    
    t=DataSet.get('tau_ave')
    t=t.reshape(-1)
    targets = np.asarray(t)
    targets =targets.transpose()
    #Conver to tensor
    decay = torch.from_numpy(decay)
    irf = torch.from_numpy(irf)
    targets = torch.from_numpy(targets)
    
    for i in range(decay.size()[0]):
        decay[1,:] = decay[i,:]/decay[i,:].max()
    print('Input train-data size',decay.shape)
    print('Input train-data-label size',targets.shape)
    
    
    
    torch_dataset = TensorDataset(decay,irf,targets)
    test_size = round(test_ratio * len(decay))
    train_size =  len(decay) - test_size
    train, test = random_split(torch_dataset, [train_size,test_size])

    train_set = DataLoader(dataset=train,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8)
    test_set = DataLoader(dataset=test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8)
    return train_set,test_set

def paramter_initialize(model):
    for i in model.modules():
        if isinstance(i,Conv1d):
            xavier_normal_(i.weight.data) 

#-----------------------------------------------------------------------------#
# train the model
def train_model(train_set, test_set, model,learning_rate = 1e-4, Epoch=200,
                USE_GPU = True, log_interval =200,patience=30, lr_step = 20, 
                lr_gamma = 0.8, path_dir = ''):

    # define the optimization
    criterion = MSELoss()
    optimizer =  Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_gamma, last_epoch=-1)


    # Record the loss 
   
    Train_Loss = list()
    
    Val_Loss = list()
    
    # enumerate epochs
    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(Epoch):
        print('Epoch {}/{}'.format(epoch+1, Epoch))
        
        
        for phase in ['train', 'val']:
            tic = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
                use_set = train_set
            else:
                model.eval()   # Set model to evaluating mode
                use_set = test_set
        # enumerate mini batches
            for batch_idx, (decays,irfs, targets) in enumerate(use_set):
                #running loss
                l = list()
                if USE_GPU:
                    decays,irfs, targets = decays.cuda(), irfs.cuda(),targets.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # compute the model output
                    yhat = model(decays.float(),irfs.float())
                    # calculate loss
                    loss = criterion(yhat, targets.float())
                    
                    
                
                if  phase == 'train':        
                    # credit assignment
                    loss.backward()
                    
                    # update model weights
                    optimizer.step()
                    
                    
                l.append(loss.cpu().detach().numpy())
                if (batch_idx % log_interval == 0 and phase == 'train'):
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(decays), len(use_set.dataset), 
                        100. * batch_idx / len(use_set), loss))
            if phase == 'train':
                
                Train_Loss.append(np.mean(l))
            else:
               
                Val_Loss.append(np.mean(l))
                ''' 
                monitor the validation loss. 
                Early stops the training if validation loss doesn't improve 
                after a given patience. 
                '''
                early_stopping(epoch+1, np.mean(l), model, path_dir) 
                
            
            toc = time.time()-tic
            print('\n{} Loss: {:.6f} time: {:.4f}s'.format(phase, np.mean(l), toc))
            if phase == 'val':        
                print('-' * 50)
                
        if early_stopping.early_stop: 
            print("Early stopping")
            break    
        Record = {
                  'Train_Loss':Train_Loss,
                  'Val_Loss':Val_Loss}
        scheduler.step()
        print('The learning rate is : {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 50)
    stop_time = time.time() - start_time
    
    print("Total training time: {:.2f}".format(stop_time))
    return Record
    