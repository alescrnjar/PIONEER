# https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
# https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
#import tqdm
import numpy as np
import random
import h5py
import os
from scipy import stats
import math

import pytorch_lightning as pl
#from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import loggers as pl_loggers
import InHouseCNN
#import deepstarr_model 
import deepstarr_model_with_init 
import tqdm
#import torchsummary

import wpcc_AC
import loss_for_aleatoric
import loss_for_evidential

""" 
https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py 
"""

def get_github_main_directory(reponame='DALdna'):
    currdir=os.getcwd()
    dir=''
    for dirname in currdir.split('/'):
        dir+=dirname+'/'
        if dirname==reponame: break
    return dir

def key_with_low(key_list,low):
    the_key=''
    for key in key_list:
        if key.lower()==low: the_key=key
    return the_key

from filelock import FileLock 

def get_evidential_loss(outputs, labels):
    #print(f"{outputs=}")
    mu=outputs[0] #QUIQUIURG gamma and mu are the same????
    nu=outputs[1] #QUIQUIURG nu or logvar????
    alpha=outputs[2]
    beta=outputs[3]
    lambda_=1. #QUIQUIURG is this choice ok????
    # def evidential_loss(y, mu, v, alpha, beta, lambda_): #loss_for_evidential.py
    loss=loss_for_evidential.evidential_loss(labels, mu, nu, alpha, beta, lambda_)
    return loss

#################################################################################################################

class PL_DeepSTARR(pl.LightningModule):
    # results={'lr': 0.002, 'train_max_epochs': 100, 'batch_size': 10, 'patience': 10, 'initialization': 'kaiming_uniform', 'initialize_dense': False}
    """
    DeepSTARR SI: "...using the Adam optimizer15 (learning rate = 0.002), mean squared error (MSE) as loss function, a batch size of 128, and early stopping with patience of ten epochs."
    """
    #def __init__(self, model):
    def __init__(self,
                 batch_size=128, #10, #original: 128, #20, #50, #100, #128,
                 train_max_epochs=100, #my would-be-choice: 50,
                 patience=10, #10, #100, #20, #patience=10,
                 min_delta=0.0, #min_delta=0.001,
                 #data_module=DeepSTARR_open_file(get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5'),
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5', 
                 input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/DeepSTARRdev.h5', #Originally created as: cp Orig_DeepSTARR_1dim.h5 DeepSTARRdev.h5
                 lr=0.002, #most likely: 0.001, #0.002 From Paper
                 initial_ds=True,

                 weight_decay=1e-6, #1e-6, #1e-6, #0.0, #1e-6, #Stage0 # WEIGHT DECAY: L2 penalty: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                 min_lr=0.0, #default when not present (configure_optimizer in evoaug_analysis_utils_AC.py)                                                     #DSRR
                 lr_patience=0, #1, #2 #,100, #10, #5
                 decay_factor=0.1, #0.0 #0.1, #Stage0

                 scale=0.001, # 0.001 or 0.005 according to Chandana

                 initialization='kaiming_uniform', # original: 'kaiming_normal', #AC 
                 initialize_dense=False, 

                 extra_str='',
                 ):
        super().__init__()
        self.scale=scale
        #self.model=deepstarr_model.DeepSTARR(output_dim=1) #.to(device) #goodold
        self.model=deepstarr_model_with_init.DeepSTARR(output_dim=1, initialization=initialization, initialize_dense=initialize_dense) #.to(device) #goodold
        #self.model=deepstarr_model_with_init.DeepSTARR_custom_init(output_dim=1,scale=self.scale) #.to(device)
        ##self.model=InHouseCNN.In_House_CNN(output_dim=1) #.to(device)
        self.name='DeepSTARR'
        self.task_type='single_task_regression'
        self.metric_names=['PCC','Spearman']
        self.el2n_scores_per_epoch=[] #EL2N
        self.initial_ds=initial_ds

        self.batch_size=batch_size
        self.train_max_epochs=train_max_epochs
        self.patience=patience
        self.lr=lr
        self.min_delta=min_delta #for trainer, but accessible as an attribute if needed                                                     #DSRR
        self.weight_decay=weight_decay

        #""
        self.min_lr=min_lr 
        self.lr_patience=lr_patience 
        self.decay_factor=decay_factor 
        #""

        self.input_h5_file=input_h5_file
        data = h5py.File(input_h5_file, 'r')
        #if input_h5_file==get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5':
        if initial_ds:
            self.X_train = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
            self.y_train = torch.tensor(np.array(data['Y_train']))[:,0].unsqueeze(1)
            self.X_test = torch.tensor(np.array(data['X_test']))
            self.y_test = torch.tensor(np.array(data['Y_test']))[:,0].unsqueeze(1)
            self.X_valid = torch.tensor(np.array(data['X_valid']))
            self.y_valid = torch.tensor(np.array(data['Y_valid']))[:,0].unsqueeze(1)                                                     #DSRR
            self.X_test2 = self.X_test # QUIQUIURG this should be ok, but it depends on if I used initials_ds=True in places I wouldnt expect
            self.y_test2 = self.y_test # QUIQUIURG this should be ok, but it depends on if I used initials_ds=True in places I wouldnt expect
        else:
            self.X_train=data['X_train']
            self.y_train=data['Y_train']
            self.X_test=data['X_test']
            self.y_test=data['Y_test']
            self.X_test2=data['X_test2']
            self.y_test2=data['Y_test2']
            self.X_valid=data['X_valid']
            self.y_valid=data['Y_valid']

    #""
    def training_step(self, batch, batch_idx): #QUIQUIURG
        #print("\n\n\n--- --- --- --- HERE HERE HERE")
        self.model.train() # https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)                                                     #DSRR

        #print(f"AC DEBUG: {torch.abs(outputs-labels).shape=}")
        if len(batch)==self.batch_size: #QUIQUIURG this is an approximation
            self.el2n_scores_per_epoch.append(np.array(torch.abs(outputs-labels).detach().cpu())) #EL2N

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #DSRR

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #DSRR
        #return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #DSRR
        out_cpu=outputs.detach().cpu()
        lab_cpu=labels.detach().cpu()
        pcc=torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC']) # QUIQUIURG is this per batch? Or having self.log on_epoch=True makes it an average of all batches? Maybe see a posteriori? See here maybe: https://www.exxactcorp.com/blog/Deep-Learning/advanced-pytorch-lightning-using-torchmetrics-and-lightning-flash
        self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     
        #return loss

    def test_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #DSRR
        #return loss
    
    def metrics(self, y_score, y_true):
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #DSRR
        spearmanr_vals=np.array(vals)
        #
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #DSRR
        pearsonr_vals=np.array(vals)
        metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals}
        return metrics

    def get_el2n_scores(self): #EL2N
        ##if 'EL2N' in model.metric_names: 
        ##if model.calc_el2n==True:      
        ##    print(f"{model.el2n_scores_per_epoch.shape=}") 
        ##    el2n_scores=np.mean(model.el2n_scores_per_epoch,axis=1)   
        #for el in self.el2n_scores_per_epoch:
        #    print(f"AC DEBUG: {el.shape=} {len(self.el2n_scores_per_epoch)=}")
        el2n_scores=np.array(self.batch_size*[0.0]) #np.mean(self.el2n_scores_per_epoch,axis=1)
        return el2n_scores

    def get_el2n_scores_with_load(self, path): #EL2N
        el2nscores_per_epoch=np.load(path)
        el2n_scores=np.mean(el2nscores_per_epoch,axis=1) #EL2N
        return el2n_scores #EL2N


    def forward(self, x):                                                     #DSRR
        return self.model(x)

    def predict_custom(self, X, keepgrad=False): #QUIQUINONURG probably there is a non custom version of this
        self.model.eval()
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)                                                     #DSRR
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        #preds=np.empty((0,1)) #GOODOLD
        #for x in dataloader:
        #print('in custom')
        #print('preds.shape: ',preds.shape)
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        return preds

    def predict_custom_mcdropout(self, X,seed=41, keepgrad=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)                                                     #DSRR
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): #QUIQUINONURG endswith ropout???
                m.train()
        #
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds







#################################################################################################################

import ResidualBind_PyTorch_AC

class PL_ResidualBind(pl.LightningModule):
    def __init__(self,
                 
                 #rbp_index=0,
                 dataset_name='VTS1',

                 batch_size=100, #V
                 train_max_epochs=300, #V
                 patience=20, #V
                 min_delta=0.0, #V
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2013.h5', #V
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2009.h5', #V
                 input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2009_processed_for_dal.h5', #V
                 lr=0.001, #V
                 initial_ds=True,

                 weight_decay=0.0, #1e-6, 
                 min_lr=0.0, #0.0, #default when not present (configure_optimizer in evoaug_analysis_utils_AC.py)                                                     #ResidualBind
                 lr_patience=7, #?
                 decay_factor=0.3, #?

                 scale=0.001, # 0.001 or 0.005 according to Chandana

                 initialization='kaiming_uniform', # original: 'kaiming_normal', #AC 
                 initialize_dense=False, 

                 extra_str='',
                 ):
        #         resnet.fit(train, valid, num_epochs=300, batch_size=100, patience=20, lr=0.001, lr_decay=0.3, decay_patience=7)
        super().__init__()
        self.scale=scale

        normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
        ss_type = 'seq' # 'seq', 'pu', or 'struct'

        self.input_h5_file=input_h5_file
        data = h5py.File(input_h5_file, 'r')
        #if input_h5_file==get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5':

        self.initial_ds=initial_ds

        """
        if initial_ds:
            #""
            #self.X_train = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
            #self.y_train = torch.tensor(np.array(data['Y_train']))[:,0].unsqueeze(1)
            #self.X_test = torch.tensor(np.array(data['X_test']))
            #self.y_test = torch.tensor(np.array(data['Y_test']))[:,0].unsqueeze(1)
            #self.X_valid = torch.tensor(np.array(data['X_valid']))
            #self.y_valid = torch.tensor(np.array(data['Y_valid']))[:,0].unsqueeze(1)                                                     #ResidualBind
            #self.X_test2 = self.X_test # QUIQUIURG this should be ok, but it depends on if I used initials_ds=True in places I wouldnt expect
            #self.y_test2 = self.y_test # QUIQUIURG this should be ok, but it depends on if I used initials_ds=True in places I wouldnt expect
            #""
            #train, valid, test = ResidualBind_PyTorch_AC.load_rnacompete_data(input_h5_file, ss_type=ss_type, normalization=normalization, rbp_index=rbp_index) # https://data.mendeley.com/datasets/m2yzh6ktzb/1 #2013
            train, valid, test = ResidualBind_PyTorch_AC.load_rnacompete_data(input_h5_file, ss_type=ss_type, normalization=normalization, dataset_name=dataset_name) # https://data.mendeley.com/datasets/m2yzh6ktzb/1 #2009

            # train['inputs'].shape=(108227, 41, 4) train['targets'].shape=(108227, 1)
            
            print(f"DEBUG {train['inputs'].shape=} {train['targets'].shape=}")

            #self.X_train=torch.tensor(np.array(train['inputs'])) #there already is a transpose in load_rnacompete_data. But it may be necessary for keras but not for torch!
            self.X_train=torch.tensor(np.transpose(np.array(train['inputs']), (0, 2, 1)))
            #self.X_train=torch.tensor(np.transpose(np.array(train['inputs']), (0, 1, 2)))
            self.y_train=torch.tensor(np.array(train['targets']))
            #
            #self.X_test=torch.tensor(np.array(test['inputs']))
            self.X_test=torch.tensor(np.transpose(np.array(test['inputs']), (0, 2, 1)))                                            #ResidualBind
            #self.X_test=torch.tensor(np.transpose(np.array(test['inputs']), (0, 1, 2)))
            self.y_test=torch.tensor(np.array(test['targets']))
            #
            #self.X_valid=torch.tensor(np.array(valid['inputs']))
            self.X_valid=torch.tensor(np.transpose(np.array(valid['inputs']), (0, 2, 1)))
            #self.X_valid=torch.tensor(np.transpose(np.array(valid['inputs']), (0, 1, 2)))
            self.y_valid=torch.tensor(np.array(valid['targets']))
            #
            self.X_test2=self.X_test
            self.y_test2=self.y_test

            input_shape = list(train['inputs'].shape)[1:]
        else:
            self.X_train=data['X_train']
            self.y_train=data['Y_train']
            self.X_test=data['X_test']
            self.y_test=data['Y_test']
            self.X_test2=data['X_test2']
            self.y_test2=data['Y_test2']                                            #ResidualBind
            self.X_valid=data['X_valid']
            self.y_valid=data['Y_valid']

            #input_shape = self.X_train[0].shape
            #input_shape = self.X_train.shape[1:]
            input_shape = (self.X_train.shape[-2],self.X_train.shape[-1])
        """
        """
        ##os.system('date')
        self.X_train=data['X_train']
        self.y_train=data['Y_train']
        self.X_test=data['X_test']
        self.y_test=data['Y_test']
        self.X_test2=data['X_test2']
        self.y_test2=data['Y_test2']
        self.X_valid=data['X_valid']
        self.y_valid=data['Y_valid']
        """
        if initial_ds:
            self.X_train=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
            self.y_train=torch.tensor(np.array(data['Y_train'])) ##.requires_grad_(True)                               #ResidualBind
            self.X_test=torch.tensor(np.array(data['X_test'])) ##.requires_grad_(True)
            self.y_test=torch.tensor(np.array(data['Y_test'])) ##.requires_grad_(True)
            self.X_valid=torch.tensor(np.array(data['X_valid'])) ##.requires_grad_(True)
            self.y_valid=torch.tensor(np.array(data['Y_valid'])) ##.requires_grad_(True)
            self.X_test2=self.X_test ##.requires_grad_(True)
            self.y_test2=self.y_test ##.requires_grad_(True)
        else:
            self.X_train=data['X_train'] ##.requires_grad_(True)
            self.y_train=data['Y_train'] ##.requires_grad_(True)
            self.X_test=data['X_test'] ##.requires_grad_(True)
            self.y_test=data['Y_test'] ##.requires_grad_(True)
            self.X_test2=data['X_test2'] ##.requires_grad_(True)                         #ResidualBind
            self.y_test2=data['Y_test2'] ##.requires_grad_(True)
            self.X_valid=data['X_valid'] ##.requires_grad_(True)
            self.y_valid=data['Y_valid'] ##.requires_grad_(True)

        ##os.system('date')
        #input_shape = self.X_train[0].shape
        #input_shape = self.X_train.shape[1:]
        #input_shape = (self.X_train.shape[-2],self.X_train.shape[-1])
        input_shape = (self.X_train.shape[-1],self.X_train.shape[-2])
        #input_shape = list(train['inputs'].shape)[1:]
        #print(f"DEBUG: {input_shape=} {self.X_train.shape=}")
        

        num_class = 1
        ##self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, weights_path='./weights.hdf5', with_residual=True) 
        ##self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, weights_path='./weights.hdf5', with_residual=False)                                             #ResidualBind
        ##os.system('date')
        #self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=True)    #first good                                          #ResidualBind
        self.model=eval('ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=True '+extra_str+')')
        #self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=False)
        ##os.system('date')

        self.name='ResidualBind'
        self.task_type='single_task_regression'
        #self.metric_names=['PCC','Spearman']
        self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 
        ##self.calc_el2n=True #EL2N
        self.el2n_scores_per_epoch=[] #EL2N

        self.batch_size=batch_size
        self.train_max_epochs=train_max_epochs
        self.patience=patience
        self.lr=lr
        self.min_delta=min_delta #for trainer, but accessible as an attribute if needed                                                     #ResidualBind
        self.weight_decay=weight_decay

        #""
        self.min_lr=min_lr 
        self.lr_patience=lr_patience 
        self.decay_factor=decay_factor 
        #""
        self.has_aleatoric='no'

    #""
    def training_step(self, batch, batch_idx): #QUIQUIURG
        #print("\n\n\n--- --- --- --- HERE HERE HERE")
        self.model.train() # https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)                                                     #ResidualBind

        """
        abs_diff = np.abs(predictions-y_train_np)
        el2n_final = np.mean(abs_diff,axis=1,keepdims=True)
        """
        self.el2n_scores_per_epoch.append(np.array(torch.abs(outputs-labels).detach().cpu())) #EL2N

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #ResidualBind

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #ResidualBind
        #return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #ResidualBind
        out_cpu=outputs.detach().cpu()
        lab_cpu=labels.detach().cpu()
        pcc=torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC']) # QUIQUIURG is this per batch? Or having self.log on_epoch=True makes it an average of all batches? Maybe see a posteriori? See here maybe: https://www.exxactcorp.com/blog/Deep-Learning/advanced-pytorch-lightning-using-torchmetrics-and-lightning-flash
        self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     
        #return loss

    def test_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #ResidualBind
        #return loss
    
    def metrics(self, y_score, y_true):
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #ResidualBind
        spearmanr_vals=np.array(vals)
        #
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #ResidualBind
        pearsonr_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.wpearsonr(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        wpcc_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20, no_weights=True))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        mse_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        wmse_vals=np.array(vals)

        #metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals}
        metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals,'WPCC':wpcc_vals,'MSE':mse_vals,'WMSE':wmse_vals}
        return metrics

    def get_el2n_scores(self): #EL2N
        ##if 'EL2N' in model.metric_names: 
        ##if model.calc_el2n==True:      
        ##    print(f"{model.el2n_scores_per_epoch.shape=}") 
        ##    el2n_scores=np.mean(model.el2n_scores_per_epoch,axis=1)   
        el2n_scores=np.mean(self.el2n_scores_per_epoch,axis=1)
        return el2n_scores

    def get_el2n_scores_with_load(self, path): #EL2N
        el2nscores_per_epoch=np.load(path)
        el2n_scores=np.mean(el2nscores_per_epoch,axis=1) #EL2N
        return el2n_scores #EL2N

    def forward(self, x):                                                     #ResidualBind
        return self.model(x)

    def predict_custom(self, X, keepgrad=False): #QUIQUINONURG probably there is a non custom version of this
        self.model.eval()
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)                                                     #ResidualBind
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        #preds=np.empty((0,1)) #GOODOLD
        #for x in dataloader:
        #print('in custom')
        #print('preds.shape: ',preds.shape)
        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        return preds

    def predict_custom_mcdropout(self, X,seed=41, keepgrad=False):
        torch.manual_seed(seed) #QUIQUIURG this has been a problem for the line defining index in test_hessian_proposer, may be also for DAL.py
        random.seed(seed)
        np.random.seed(seed)                                                     #ResidualBind
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): #QUIQUINONURG endswith ropout???
                m.train()
        #
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds



from PL_mpra import *




# HUMAN
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

import LegNetPK  #HUMAN
#https://github.com/autosome-ru/human_legnet/blob/main/trainer.py
#https://colab.research.google.com/drive/1FDJGP55xscybfdbrS25YkHxjGzW3Lppm?usp=sharing#scrollTo=N1ZoIlVTJQ_v
class PL_LegNetPK(pl.LightningModule): 
    def __init__(self,
                 
                 #lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5
                 dataset_name='LentiMPRA',

                 batch_size=100, #64, #V
                 #batch_size=128,
                 #batch_size=64,
                 train_max_epochs=100, #V
                 patience=10, #V
                 min_delta=0, #V?
                 input_h5_file=get_github_main_directory(reponame='Occasio_Dev')+'inputs/LentiMPRA_processed_for_dal.h5', #V
                 lr=0.001, #V
                 initial_ds=True,

                 weight_decay=1e-6, #V
                 min_lr=0., #V?
                 lr_patience=5, #V
                 decay_factor=0.1, #V

                 scale=0.001, #V

                 initialization='kaiming_uniform', # original: 'kaiming_normal', #AC 
                 initialize_dense=False, 

                 extra_str='',
                 ):
        super().__init__()
        self.scale=scale

        self.input_h5_file=input_h5_file
        data = h5py.File(input_h5_file, 'r')
        #if input_h5_file==get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5':

        self.initial_ds=initial_ds

        if initial_ds:
            self.X_train=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
            self.y_train=torch.tensor(np.array(data['Y_train'])) ##.requires_grad_(True)                               #Lenti
            self.X_test=torch.tensor(np.array(data['X_test'])) ##.requires_grad_(True)
            self.y_test=torch.tensor(np.array(data['Y_test'])) ##.requires_grad_(True)
            self.X_valid=torch.tensor(np.array(data['X_valid'])) ##.requires_grad_(True)
            self.y_valid=torch.tensor(np.array(data['Y_valid'])) ##.requires_grad_(True)
            self.X_test2=self.X_test ##.requires_grad_(True)
            self.y_test2=self.y_test ##.requires_grad_(True)
        else:
            self.X_train=data['X_train'] ##.requires_grad_(True)
            self.y_train=data['Y_train'] ##.requires_grad_(True)
            self.X_test=data['X_test'] ##.requires_grad_(True)
            self.y_test=data['Y_test'] ##.requires_grad_(True)
            self.X_test2=data['X_test2'] ##.requires_grad_(True)                         #Lenti
            self.y_test2=data['Y_test2'] ##.requires_grad_(True)
            self.X_valid=data['X_valid'] ##.requires_grad_(True)
            self.y_valid=data['Y_valid'] ##.requires_grad_(True)

        ##os.system('date')
        #input_shape = self.X_train[0].shape
        #input_shape = self.X_train.shape[1:]
        #input_shape = (self.X_train.shape[-2],self.X_train.shape[-1])
        input_shape = (self.X_train.shape[-1],self.X_train.shape[-2])
        #input_shape = list(train['inputs'].shape)[1:]
        #print(f"DEBUG: {input_shape=} {self.X_train.shape=}")
        

        num_class = 1
        ##self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, weights_path='./weights.hdf5', with_residual=True) 
        ##self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, weights_path='./weights.hdf5', with_residual=False)                                             #ResidualBind
        ##os.system('date')
        #self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=True)    #first good                                          #ResidualBind
        #self.model=eval('ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=True '+extra_str+')')
        #self.model=ResidualBind_PyTorch_AC.ResidualBind_AC(input_shape, num_class, with_residual=False)
        ##os.system('date')
        #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model=LegNetPK.LegNetPK(4) #.to(device)
        #self.device=device

        if 'heteroscedastic' in extra_str:
            self.has_aleatoric='heteroscedastic'
            self.metric_names=['PCC','Spearman','MSE',
                               'PCCaleat','Spearmanaleat','MSEaleat'] 
        elif 'evidential' in extra_str:
            self.has_aleatoric='evidential'
            self.metric_names=['PCC']
        else:
            self.has_aleatoric='no'
            #self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 
            self.metric_names=['PCC','Spearman','MSE']

        if self.has_aleatoric=='no':
            self.model=LegNetPK.LegNetPK(4) #.to(device)
        elif self.has_aleatoric=='heteroscedastic':
            self.model=LegNetPK.LegNetPK(in_ch=4,unc_control='heteroscedastic') #.to(device)
        elif self.has_aleatoric=='evidential':
            self.model=LegNetPK.LegNetPK(in_ch=4,unc_control='evidential') #.to(device)

        self.name='LegNet'
        self.task_type='single_task_regression'
        #self.metric_names=['PCC','Spearman']
        #self.metric_names=['PCC','Spearman','MSE'] #,'WPCC','MSE','WMSE'] 
        ##self.calc_el2n=True #EL2N
        self.el2n_scores_per_epoch=[] #EL2N

        self.batch_size=batch_size
        self.train_max_epochs=train_max_epochs
        self.patience=patience
        self.lr=lr
        self.min_delta=min_delta #for trainer, but accessible as an attribute if needed                                                     #Lenti
        self.weight_decay=weight_decay

        #""
        self.min_lr=min_lr 
        self.lr_patience=lr_patience 
        self.decay_factor=decay_factor 
        #""

    #""
    def training_step(self, batch, batch_idx): #QUIQUIURG
        #print("\n\n\n--- --- --- --- HERE HERE HERE")
        self.model.train() # https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
        elif self.has_aleatoric=='heteroscedastic': 
            loss= loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)  
        #""
        #abs_diff = np.abs(predictions-y_train_np)
        #el2n_final = np.mean(abs_diff,axis=1,keepdims=True)
        #""
        if self.has_aleatoric=='no':
            self.el2n_scores_per_epoch.append(np.array(torch.abs(outputs-labels).detach().cpu())) #EL2N

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #Lenti

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #ResidualBind
        #return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
        elif self.has_aleatoric=='heteroscedastic': 
            loss= loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)  
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #Lenti
        if self.has_aleatoric=='no':
            out_cpu=outputs.detach().cpu()
            lab_cpu=labels.detach().cpu()
            pcc=torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC']) # QUIQUIURG is this per batch? Or having self.log on_epoch=True makes it an average of all batches? Maybe see a posteriori? See here maybe: https://www.exxactcorp.com/blog/Deep-Learning/advanced-pytorch-lightning-using-torchmetrics-and-lightning-flash
            self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     
        #return loss

    def test_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
        elif self.has_aleatoric=='heteroscedastic': 
            loss= loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)  
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #Lenti
        #return loss
    
    def metrics(self, y_score, y_true):
        if self.has_aleatoric!='evidential':
            vals = []
            for output_index in range(y_score.shape[-1]):
                #print(f"{output_index=}")
                #print(f"{y_true.shape=}")
                #print(f"{y_score.shape=}")
                vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #Lenti
            spearmanr_vals=np.array(vals)
            #
            vals = []
            for output_index in range(y_score.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #Lenti
            pearsonr_vals=np.array(vals)

            #vals = []
            #for output_index in range(y_score.shape[-1]):
            #    vals.append(wpcc_AC.wpearsonr(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
            #wpcc_vals=np.array(vals)

            vals = []
            for output_index in range(y_score.shape[-1]):
                vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20, no_weights=True))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
            mse_vals=np.array(vals)

            #vals = []
            #for output_index in range(y_score.shape[-1]):
            #    vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
            #wmse_vals=np.array(vals)

            #metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals}
            metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals,'MSE':mse_vals} #,'WPCC':wpcc_vals,'WMSE':wmse_vals}
        else:
            yscore=y_score[0] #QUIQUIURG
            vals = []
            #print(f"{yscore=}")
            if (yscore==math.nan).all():
                yscore=np.ones(y_score.shape)
            for output_index in range(yscore.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,output_index], yscore[:,output_index])[0] )        
            pearsonr_vals=np.array(vals)
            metrics={'PCC':pearsonr_vals}
        return metrics

    def get_el2n_scores(self): #EL2N
        ##if 'EL2N' in model.metric_names: 
        ##if model.calc_el2n==True:      
        ##    print(f"{model.el2n_scores_per_epoch.shape=}") 
        ##    el2n_scores=np.mean(model.el2n_scores_per_epoch,axis=1)   
        el2n_scores=np.mean(self.el2n_scores_per_epoch,axis=1)
        return el2n_scores

    def get_el2n_scores_with_load(self, path): #EL2N
        el2nscores_per_epoch=np.load(path)
        el2n_scores=np.mean(el2nscores_per_epoch,axis=1) #EL2N
        return el2n_scores #EL2N

    def forward(self, x):                                                     #Lenti
        return self.model(x)

    def predict_custom(self, X, keepgrad=False): #QUIQUINONURG probably there is a non custom version of this
        self.model.eval()
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)                                                     #Lenti
        if self.has_aleatoric!='evidential':
            preds=torch.empty(0)
            if keepgrad: 
                preds = preds.to(self.device)
            else:
                preds = preds.cpu()
        else:
            preds0=torch.empty(0)
            preds1=torch.empty(0)
            preds2=torch.empty(0)
            preds3=torch.empty(0)
            if keepgrad: 
                preds0 = preds0.to(self.device)
                preds1 = preds1.to(self.device)
                preds2 = preds2.to(self.device)
                preds3 = preds3.to(self.device)
            else:
                preds0 = preds0.cpu()
                preds1 = preds1.cpu()
                preds2 = preds2.cpu()
                preds3 = preds3.cpu()
            
        #preds=np.empty((0,1)) #GOODOLD
        #for x in dataloader:
        #print('in custom')
        #print('preds.shape: ',preds.shape)
        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if self.has_aleatoric=='no' or self.has_aleatoric=='heteroscedastic':
                if not keepgrad: pred=pred.detach().cpu()
            elif self.has_aleatoric=='evidential':
                if not keepgrad: 
                    pred=pred[0].detach().cpu(),pred[1].detach().cpu(),pred[2].detach().cpu(),pred[3].detach().cpu()
            if self.has_aleatoric!='evidential':
                preds=torch.cat((preds,pred),axis=0)
            else:
                #print(f"NOOOOOOAJ {len(preds)=} {preds=} {len(pred)=} {pred=}")
                #exit()
                preds0=torch.cat((preds0,pred[0]),axis=0)
                preds1=torch.cat((preds1,pred[1]),axis=0)
                preds2=torch.cat((preds2,pred[2]),axis=0)
                preds3=torch.cat((preds3,pred[3]),axis=0)
        if self.has_aleatoric=='evidential':
            preds=preds0,preds1,preds2,preds3
        return preds

    def predict_custom_mcdropout(self, X,seed=41, keepgrad=False):
        torch.manual_seed(seed) #QUIQUIURG this has been a problem for the line defining index in test_hessian_proposer, may be also for DAL.py
        random.seed(seed)
        np.random.seed(seed)                                                     #Lenti
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): #QUIQUINONURG endswith ropout???
                m.train()
        #
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds

###########################################


import newresnet

class PL_NewResNet(pl.LightningModule):
    def __init__(self,
                 
                 #rbp_index=0,
                 dataset_name='VTS1',

                 batch_size=100, #V
                 train_max_epochs=300, #V
                 patience=20, #V
                 min_delta=0.0, #V
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2013.h5', #V
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2009.h5', #V
                 #input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/rnacompete2009_processed_for_dal.h5', #V
                 input_h5_file=get_github_main_directory(reponame='DALdna')+'inputs/newLentiMPRAHepG2_processed_for_dal.h5', #V
                 lr=0.001, #V
                 initial_ds=True,

                 weight_decay=0.0, #1e-6, 
                 min_lr=0.0, #0.0, #default when not present (configure_optimizer in evoaug_analysis_utils_AC.py)                                                     #NewResNet
                 lr_patience=7, #?
                 decay_factor=0.3, #?

                 scale=0.001, # 0.001 or 0.005 according to Chandana

                 initialization='kaiming_uniform', # original: 'kaiming_normal', #AC 
                 initialize_dense=False, 

                 extra_str='',
                 ):
        #         resnet.fit(train, valid, num_epochs=300, batch_size=100, patience=20, lr=0.001, lr_decay=0.3, decay_patience=7)
        super().__init__()
        self.scale=scale

        normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
        ss_type = 'seq' # 'seq', 'pu', or 'struct'

        self.input_h5_file=input_h5_file
        data = h5py.File(input_h5_file, 'r')
        #if input_h5_file==get_github_main_directory(reponame='DALdna')+'inputs/Orig_DeepSTARR_1dim.h5':

        self.initial_ds=initial_ds

        if initial_ds:
            self.X_train=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
            self.y_train=torch.tensor(np.array(data['Y_train'])) ##.requires_grad_(True)                               #NewResNet
            self.X_test=torch.tensor(np.array(data['X_test'])) ##.requires_grad_(True)
            self.y_test=torch.tensor(np.array(data['Y_test'])) ##.requires_grad_(True)
            self.X_valid=torch.tensor(np.array(data['X_valid'])) ##.requires_grad_(True)
            self.y_valid=torch.tensor(np.array(data['Y_valid'])) ##.requires_grad_(True)
            self.X_test2=self.X_test ##.requires_grad_(True)
            self.y_test2=self.y_test ##.requires_grad_(True)
        else:
            self.X_train=data['X_train'] ##.requires_grad_(True)
            self.y_train=data['Y_train'] ##.requires_grad_(True)
            self.X_test=data['X_test'] ##.requires_grad_(True)
            self.y_test=data['Y_test'] ##.requires_grad_(True)
            self.X_test2=data['X_test2'] ##.requires_grad_(True)                         #NewResNet
            self.y_test2=data['Y_test2'] ##.requires_grad_(True)
            self.X_valid=data['X_valid'] ##.requires_grad_(True)
            self.y_valid=data['Y_valid'] ##.requires_grad_(True)

        ##os.system('date')
        #input_shape = self.X_train[0].shape
        #input_shape = self.X_train.shape[1:]
        #input_shape = (self.X_train.shape[-2],self.X_train.shape[-1])
        input_shape = (self.X_train.shape[-1],self.X_train.shape[-2])
        #input_shape = list(train['inputs'].shape)[1:]
        #print(f"DEBUG: {input_shape=} {self.X_train.shape=}")
        
        if 'heteroscedastic' in extra_str:
            self.has_aleatoric='heteroscedastic'
            self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE',
                               'PCCaleat','Spearmanaleat','WPCCaleat','MSEaleat','WMSEaleat'] 
        elif 'evidential' in extra_str:
            self.has_aleatoric='evidential'
            self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 
        else:
            self.has_aleatoric='no'
            self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 

        num_class = 1
        if self.has_aleatoric=='no':
            self.model=newresnet.NewResNet()
        elif self.has_aleatoric=='heteroscedastic':
            self.model=newresnet.NewResNet(unc_control='heteroscedastic')
        elif self.has_aleatoric=='evidential':
            self.model=newresnet.NewResNet(unc_control='evidential')

        self.name='ResidualBind'
        self.task_type='single_task_regression'
        #self.metric_names=['PCC','Spearman']
        #self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 
        ##self.calc_el2n=True #EL2N
        self.el2n_scores_per_epoch=[] #EL2N

        self.batch_size=batch_size
        self.train_max_epochs=train_max_epochs
        self.patience=patience
        self.lr=lr
        self.min_delta=min_delta #for trainer, but accessible as an attribute if needed                                                     #NewResNet
        self.weight_decay=weight_decay

        #""
        self.min_lr=min_lr 
        self.lr_patience=lr_patience 
        self.decay_factor=decay_factor 
        #""


    #""
    def training_step(self, batch, batch_idx): #QUIQUIURG
        #print("\n\n\n--- --- --- --- HERE HERE HERE")
        self.model.train() # https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)                                                     #NewResNet
        elif self.has_aleatoric=='heteroscedastic': 
            loss=loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)                                                     #NewResNet

        """
        abs_diff = np.abs(predictions-y_train_np)
        el2n_final = np.mean(abs_diff,axis=1,keepdims=True)
        """
        if self.has_aleatoric=='no':
            self.el2n_scores_per_epoch.append(np.array(torch.abs(outputs-labels).detach().cpu())) #EL2N

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #NewResNet

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #NewResNet
        #return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
        elif self.has_aleatoric=='heteroscedastic': 
            loss= loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)  
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #NewResNet
        if self.has_aleatoric=='no':
            out_cpu=outputs.detach().cpu()
            lab_cpu=labels.detach().cpu()
            pcc=torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC']) # QUIQUIURG is this per batch? Or having self.log on_epoch=True makes it an average of all batches? Maybe see a posteriori? See here maybe: https://www.exxactcorp.com/blog/Deep-Learning/advanced-pytorch-lightning-using-torchmetrics-and-lightning-flash
            self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     
        #return loss

    def test_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
        elif self.has_aleatoric=='heteroscedastic': 
            loss= loss_for_aleatoric.gaussian_nll_loss(labels, outputs)
        elif self.has_aleatoric=='evidential':
            loss=get_evidential_loss(outputs, labels)  
        loss = loss_fn(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #NewResNet
        #return loss
    
    def metrics(self, y_score, y_true):
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #NewResNet
        spearmanr_vals=np.array(vals)
        #
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #NewResNet
        pearsonr_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.wpearsonr(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        wpcc_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20, no_weights=True))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        mse_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_AC.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        wmse_vals=np.array(vals)

        #metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals}
        if self.has_aleatoric=='no':
            metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals,'WPCC':wpcc_vals,'MSE':mse_vals,'WMSE':wmse_vals}
        elif self.has_aleatoric=='heteroscedastic':
            metrics={'Spearman':spearmanr_vals[0],'PCC':pearsonr_vals[0],
                     'Spearmanaleat':spearmanr_vals[1],'PCCaleat':pearsonr_vals[1],
                     'WPCC':wpcc_vals[0],'MSE':mse_vals[0],'WMSE':wmse_vals[0],
                     'WPCCaleat':wpcc_vals[1],'MSEaleat':mse_vals[1],'WMSEaleat':wmse_vals[1]}
        return metrics

    def get_el2n_scores(self): #EL2N
        ##if 'EL2N' in model.metric_names: 
        ##if model.calc_el2n==True:      
        ##    print(f"{model.el2n_scores_per_epoch.shape=}") 
        ##    el2n_scores=np.mean(model.el2n_scores_per_epoch,axis=1)   
        el2n_scores=np.mean(self.el2n_scores_per_epoch,axis=1)
        return el2n_scores

    def get_el2n_scores_with_load(self, path): #EL2N
        el2nscores_per_epoch=np.load(path)
        el2n_scores=np.mean(el2nscores_per_epoch,axis=1) #EL2N
        return el2n_scores #EL2N

    def forward(self, x):                                                     #NewResNet
        return self.model(x)

    def predict_custom(self, X, keepgrad=False): #QUIQUINONURG probably there is a non custom version of this
        self.model.eval()
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)                                                     #NewResNet
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        #preds=np.empty((0,1)) #GOODOLD
        #for x in dataloader:
        #print('in custom')
        #print('preds.shape: ',preds.shape)
        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if self.has_aleatoric=='no' or self.has_aleatoric=='heteroscedastic':
                if not keepgrad: pred=pred.detach().cpu()
            elif self.has_aleatoric=='evidential':
                if not keepgrad: 
                    pred=pred[0].detach().cpu(),pred[1].detach().cpu(),pred[2].detach().cpu(),pred[3].detach().cpu()
            if self.has_aleatoric!='evidential':
                preds=torch.cat((preds,pred),axis=0)
            else:
                preds=torch.cat((preds[0],pred[0]),axis=0),torch.cat((preds[1],pred[1]),axis=0),torch.cat((preds[2],pred[2]),axis=0),torch.cat((preds[3],pred[3]),axis=0)
        return preds

    def predict_custom_mcdropout(self, X,seed=41, keepgrad=False):
        torch.manual_seed(seed) #QUIQUIURG this has been a problem for the line defining index in test_hessian_proposer, may be also for DAL.py
        random.seed(seed)
        np.random.seed(seed)                                                     #NewResNet
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): #QUIQUINONURG endswith ropout???
                m.train()
        #
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        #for x in tqdm.tqdm(dataloader, total=len(dataloader)):
        for x in dataloader:
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds




###########################################


def training_with_PL(chosen_model, chosen_dataset,
                     initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=False,
                     extra_str='',extrastrflag='',
                     inputdir="../inputs/",
                     outdir="../../outputs_DALdna/"
                     ):

    """
    if wanted_wandb: 
        import wandb
        os.system('wandb login 7693bea9dff37d98f5d98928ab2fbe3842552e84')
        #os.system('export WANDB_API_KEY=$(cat wandb_api_key_acrnjar.txt)') #should go in the job scheduler # https://wandb.ai/acrnjar
        wandb_project = config['wandb_'+chosen_model+'_'+chosen_dataset] # project name
        wandb_name = config['wandb_ray_'+chosen_model+'_'+chosen_dataset+'_seed-'+str(myseed)] # give a run/trial name here
        wandb.init(project=wandb_project, name=wandb_name, config=config)
    """
    if wanted_wandb:
        ##os.system('export WANDB_API_KEY=$(cat wandb_api_key_acrnjar.txt)') #should go in the job scheduler # https://wandb.ai/acrnjar
        import wandb
        from pytorch_lightning.loggers import WandbLogger # https://docs.wandb.ai/guides/integrations/lightning
        wandb_logger = WandbLogger(log_model="all")

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=} {torch.cuda.is_available()=}")

    currdir=os.popen('pwd').read().replace("\n","") #os.getcwd()
    #outdir="../../outputs_DALdna/" #PERFECT OLD
    log_dir=outdir+"lightning_logs_"+chosen_model+"/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if wanted_wandb:
        logger_of_choice=wandb_logger
    else:
        logger_of_choice=tb_logger

    #if chosen_model=='InHouseCNN': 
    #    model=PL_InHouseCNN(input_h5_file='../inputs/'+chosen_dataset+'.h5', initial_ds=True)
    #if chosen_model=='DeepSTARR': 
    #    model=PL_DeepSTARR(input_h5_file='../inputs/'+chosen_dataset+'.h5', initial_ds=True)
    #model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds=True)") #PERFECT OLD
    #model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds=True, extra_str='"+extra_str+"')") #PERFECT OLD
    model=eval("PL_"+chosen_model+"(input_h5_file='"+inputdir+chosen_dataset+".h5', initial_ds=True, extra_str='"+extra_str+"')") #PERFECT OLD
    ##model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds=True)").to(device)

    #model=model.to(device)

    #if verbose: torchsummary.summary(model, model.X_train[0].shape)

    os.system('date')
    print("Loading training dataset...")
    train_dataloader=torch.utils.data.DataLoader(list(zip(model.X_train,model.y_train)), batch_size=model.batch_size, shuffle=True)
    os.system('date')
    print("Loading valid dataset...")
    valid_dataloader=torch.utils.data.DataLoader(list(zip(model.X_valid,model.y_valid)), batch_size=model.batch_size, shuffle=True) #True)
    os.system('date')
    print("Loading test dataset...")
    test_dataloader=torch.utils.data.DataLoader(list(zip(model.X_test,model.y_test)), batch_size=model.batch_size, shuffle=True) #True)
    os.system('date')

    if extrastrflag=='':
        ckptfile="oracle_"+model.name+"_"+chosen_dataset #+".ckpt" #perfect old
    else:
        ckptfile="oracle_"+model.name+"_"+chosen_dataset+"_"+extrastrflag #+".ckpt"
    to_monitor='val_loss' #QUIQUIURG for callback accuracy you should have defined self.log('acc',acc_val) and a whole method for calculating it within that function. https://lightning.ai/docs/pytorch/stable/extensions/logging.html
    callback_ckpt = pl.callbacks.ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
        #gpus=1,
        #auto_select_gpus=True,
        monitor=to_monitor, #default is None which saves a checkpoint only for the last epoch.
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=inputdir, #get_github_main_directory(reponame='DALdna')+"inputs/", 
        filename=ckptfile, #comment out to verify that a different epoch is picked in the name.
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
                            #monitor='val_loss',
                            monitor=to_monitor,
                            min_delta=model.min_delta, #https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
                            patience=model.patience,
                            verbose=False,
                            mode='min'
                            )

    if initial_test:
        print('predict_custom')
        y_score=model.predict_custom(model.X_test)
        print('y_score.shape: ',y_score.shape)
        metrics_pretrain=model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")
    
    if mcdropout_test:
        n_mc = 5
        preds_mc=torch.zeros((n_mc,len(model.X_test)))
        for i in range(n_mc):
            preds_mc[i] = model.predict_custom_mcdropout(model.X_test,
                                                            seed=41+i).squeeze(axis=1).unsqueeze(axis=0)
        print('predict_custom_mcdropout')
        print('y_score.shape: ',preds_mc.shape)
        metrics_pretrain=model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")

    print(f"{model.device=}")
    #trainer = pl.Trainer(max_epochs=model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt,early_stop_callback],deterministic=True) #, val_check_interval=1) #callbacks=["ADD CALLBACKS", callback_model]) #, logger=None) #,patience=patience) #PERFECT OLD
    trainer = pl.Trainer(accelerator='cuda', devices=-1, max_epochs=model.train_max_epochs, logger=logger_of_choice, callbacks=[callback_ckpt,early_stop_callback],deterministic=True) #, val_check_interval=1) #callbacks=["ADD CALLBACKS", callback_model]) #, logger=None) #,patience=patience) # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) #GOODOLD
    #os.system('mv ../inputs/'+ckptfile+'-v1.ckpt ../inputs/'+ckptfile+'.ckpt')
    os.system('mv '+inputdir+ckptfile+'-v1.ckpt '+inputdir+ckptfile+'.ckpt')
    #if model.has_aleatoric!='evidential':
    if verbose: os.system('date')
    y_score=model.predict_custom(model.X_test)
    if verbose: os.system('date')
    metrics=model.metrics(y_score, model.y_test)


    ##if 'EL2N' in model.metric_names: #EL2N
    ##if model.calc_el2n==True:      #EL2N
    ##    print(f"{model.el2n_scores_per_epoch.shape=}") #EL2N
    ##    el2n_scores=np.mean(model.el2n_scores_per_epoch,axis=1)   #EL2N

    print(metrics)

    """
    if wanted_wandb:
        wandb.log(metrics)
    """

    print(ckptfile)
    return metrics











##############################################################





if __name__=='__main__':
    #import sys
    #sys.path.append('./src')
    #import evoaug_AC


    caselist=[
              #['InHouseCNN', 'ATF2_200'], 
              #['InHouseCNN', 'TIA1_K562_200'],  

              #['InHouseCNN', 'RBFOX2_K562_200'],
              #['InHouseCNN', 'HNRNPK_K562_200'], 
              #['InHouseCNN', 'PTBP1_K562_200'],
              #['InHouseCNN', 'PUM2_K562_200'],  
              #['InHouseCNN', 'QKI_K562_200'], 
              #['InHouseCNN', 'SF3B4_K562_200'], 
              #['InHouseCNN', 'SRSF1_K562_200'],  
              #['InHouseCNN', 'TARDBP_K562_200'],  
              #['InHouseCNN', 'U2AF1_K562_200'],

              #['InHouseCNN', 'pseudolabels_ATF2_200'], #from PL_Models_infere.py
              
              #['DeepSTARR', 'DeepSTARRdev', '', '1'],
              #['DeepSTARR', 'DeepSTARRhk'], #QUIQUIURG this has higher PCC! 0.75!

              #['Daedalus', 'DeepSTARRdev'],

              #['ResidualBind','rnacompete2009'],
              #['ResidualBind','rnacompete2009_processed_for_dal', '', '']
              #['ResidualBind','rnacompete2009_processed_for_dal', ', wanted_initial_attention=True', 'initAtt']
              ##['ResidualBind','rnacompete2009_processed_for_dal', ', wanted_hook=True', '_withstoredgrad']
              #['ResidualBind','rnacompete2009_processed_for_dal', ', activation="softplus_beta1"', 'softplus'] #goodold
              #
              # GOODOLD used until 21 Feb 2024
              #['ResidualBind','VTS1_rnacompete2009_processed_for_dal', ', activation="softplus_beta1"', 'softplus'] #
              #['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="softplus_beta1"', 'softplus'] 
              #    def __init__(self, input_shape=(41,4), num_class=1, classification=False, with_residual=True, activation='ReLU', wanted_initial_attention=False, wanted_hook=False):

              # DEFINITIVE?
              ##['ResidualBind','VTS1_rnacompete2009_processed_for_dal', ', activation="ReLU"', 'relustandard'] # ReLU not relu: checked, its ok 
              ##['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard'] 
              #['ResidualBind','VTS1_rnacompete2009_processed_for_dal', ', activation="ReLU"', 'relustandard1']  # with overlooked ReLU
              #['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard1'] #will be the definitive (Credo)
              
              # DEFINITIVE FOR HESSIAN?
              #['ResidualBind','VTS1_rnacompete2009_processed_for_dal', ', activation="softplus_beta1"', 'softplusagain'] #changed only "softplusagain", but requires addressing lines with TURESOFT before launching
              #['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="softplus_beta1"', 'softplusagain'] 

              ##['LegNet','complexmediaHQtestdata_processed_for_dal', '','']
              #['LegNet','complex-media-training-data-Glu_Native-complex-noheaderlabels-seed0_random0_5000','',''] #LAST ATTEMPTED
              #['ResidualBind','LentiMPRA_processed_for_dal',', activation="ReLU"', 'relustandard1']
              #['ResNetLenti','LentiMPRA_processed_for_dal','', '']
              #['LegNet','LentiMPRA_processed_for_dal','', 'bs100']
              #['LegNet','LentiMPRA_processed_for_dal','', 'bs128']
              #['LegNet','LentiMPRA_processed_for_dal','', 'bs64']
              #['ResidualBind','DeepSTARRdev',', activation="ReLU"', 'relustandard1']
              #['ResidualBind','RBFOX1_rnacompete2013labels-seed0_random0_5000',', activation="ReLU"', 'relustandard1_test']
              #['ResidualBind','prov_dal_dataset_pristine_seed-0-0',', activation="ReLU"', 'relustandard1_test']

             #['ResBindModified','LentiMPRA_processed_for_dal','', '']
             #['ResidualBind','LentiMPRA_processed_for_dal',', activation="ReLU"', 'prov']
            #['mpra','LentiMPRA_processed_for_dal','', 'prov']
            #['LegNet','LentiMPRA_labels-seed0_random0_5000','', 'prov']
            #['LegNet','LentiMPRA_processed_for_dal','', 'prov'] #0.6...
            #['LegNetPK','LentiMPRA_processed_for_dal','', 'prov'] #0.73
            #['LegNetPK','prov_HepG2tsv_processed_for_dal','', '']
            
            #['LegNetPK','HepG2tsv_processed_for_dal','', ''], # 0.999...
            #['LegNetPK','HepG2tsv_processed_for_dal','', '1'], #0.9972
            #['LegNetPK','HepG2tsv_processed_for_dal','', '2'], 
            #['LegNetPK','complex_media_3929-3929_processed_for_dal','', '3929-3929'],
            #['LegNet','HepG2tsv_processed_for_dal','', ''],
            #['LegNet','complex_media_3929-3929_processed_for_dal','', '3929-3929'],

            #['NewResNet','newLentiMPRAK562two_processed_for_dal','heteroscedastic', 'heteroscedastic'], # {'Spearman': 0.6869097282693476, 'PCC': 0.7563271842360576, 'Spearmanaleat': 0.0948490436952949, 'PCCaleat': 0.09905171674147858, 'WPCC': 0.758665921427769, 'MSE': 0.1707406536001626, 'WMSE': 0.17074406, 'WPCCaleat': 0.09974320715300665, 'MSEaleat': 5.665717457178491, 'WMSEaleat': 5.668789}                                      ### 0.7152162029331273,

            #['NewResNet','newLentiMPRAK562_labels-seed0_random0_5000','', ''], #0.322            
            
            #['NewResNet','newLentiMPRAK562two_labels-seed0_random0_5000','heteroscedastic', 'heteroscedastic'], # {'Spearman': 0.220317330373988, 'PCC': 0.09211046302707393, 'Spearmanaleat': 0.1255393825647983, 'PCCaleat': 0.037075800606756526, 'WPCC': 0.08895881538198619, 'MSE': 0.31949661399084195, 'WMSE': 0.3195439, 'WPCCaleat': 0.037778316995375615, 'MSEaleat': 2.9856317407478667, 'WMSEaleat': 2.989346}    ###0.21855

            #['LegNetPK','newLentiMPRAK562_processed_for_dal','evidential', 'evidential'],

            #['ResidualBind','VTS1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard1']
            #['ResidualBind','VTS1_rnacompete2009_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','VTS1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','RNCMPT00001_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','RNCMPT00082_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','RNCMPT00002_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']
            #['ResidualBind','RNCMPT00003_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'prov']

            ['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard151'],
            ['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard152'],
            ['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard153'],
            ['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard154'],
            ['ResidualBind','RBFOX1_rnacompete2013_processed_for_dal', ', activation="ReLU"', 'relustandard155'],
    ] 

    for case in caselist:
        print("-------------- CASE:",case)

        chosen_model=case[0]
        chosen_dataset=case[1]
        extra_str=case[2]
        extrastrflag=case[3]

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        overall_seed=41
        if extrastrflag=='relustandard151': overall_seed=51
        if extrastrflag=='relustandard152': overall_seed=52
        if extrastrflag=='relustandard153': overall_seed=53
        if extrastrflag=='relustandard154': overall_seed=54
        if extrastrflag=='relustandard155': overall_seed=55
        myseed=overall_seed
        torch.manual_seed(myseed)
        random.seed(myseed)
        np.random.seed(myseed)

        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        #metrics=training_with_PL(chosen_model, chosen_dataset, initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=True, extra_str=extra_str,extrastrflag=extrastrflag)
        os.system('date')
        metrics=training_with_PL(chosen_model, chosen_dataset, initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=False, extra_str=extra_str,extrastrflag=extrastrflag)
        os.system('date')
        print("case:",case,metrics)
        
        print("SCRIPT END")
        print("WARNING: should I do a deep ensemble, and then take the ckpt of the best model?")
