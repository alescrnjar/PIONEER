from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import h5py
import os
from scipy import stats
import math

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import deepstarr_custom
import tqdm

import wpcc_utils

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

#################################################################################################################

class PL_DeepSTARR(pl.LightningModule):
    """
    DeepSTARR SI: "...using the Adam optimizer15 (learning rate = 0.002), mean squared error (MSE) as loss function, a batch size of 128, and early stopping with patience of ten epochs."
    """
    def __init__(self,
                 batch_size=128, #10, #original: 128, #20, #50, #100, #128,
                 train_max_epochs=100, #my would-be-choice: 50,
                 patience=10, #10, #100, #20, #patience=10,
                 min_delta=0.0, #min_delta=0.001,
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
        self.model=deepstarr_custom.DeepSTARR(output_dim=1, initialization=initialization, initialize_dense=initialize_dense) #.to(device) #goodold

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

import ResidualBind_PyTorch

class PL_ResidualBind(pl.LightningModule):
    def __init__(self,
                 
                 #rbp_index=0,
                 dataset_name='VTS1',

                 batch_size=100, #V
                 train_max_epochs=300, #V
                 patience=20, #V
                 min_delta=0.0, #V

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
        super().__init__()
        self.scale=scale

        normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
        ss_type = 'seq' # 'seq', 'pu', or 'struct'

        self.input_h5_file=input_h5_file
        data = h5py.File(input_h5_file, 'r')

        self.initial_ds=initial_ds


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

        input_shape = (self.X_train.shape[-1],self.X_train.shape[-2])

        num_class = 1

        self.model=eval('ResidualBind_PyTorch.ResidualBind(input_shape, num_class, with_residual=True '+extra_str+')')

        self.name='ResidualBind'
        self.task_type='single_task_regression'
        self.metric_names=['PCC','Spearman','WPCC','MSE','WMSE'] 
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
            vals.append(wpcc_utils.wpearsonr(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        wpcc_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_utils.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20, no_weights=True))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
        mse_vals=np.array(vals)

        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(wpcc_utils.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
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

        for x in dataloader:
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds


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

import LegNet_Custom  #HUMAN
class PL_LegNet_Custom(pl.LightningModule): 
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

        input_shape = (self.X_train.shape[-1],self.X_train.shape[-2])

        num_class = 1

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
            self.model=LegNet_Custom.LegNet_Custom(4) #.to(device)
        elif self.has_aleatoric=='heteroscedastic':
            self.model=LegNet_Custom.LegNet_Custom(in_ch=4,unc_control='heteroscedastic') #.to(device)
        elif self.has_aleatoric=='evidential':
            self.model=LegNet_Custom.LegNet_Custom(in_ch=4,unc_control='evidential') #.to(device)

        self.name='LegNet'
        self.task_type='single_task_regression'
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

        if self.has_aleatoric=='no':
            self.el2n_scores_per_epoch.append(np.array(torch.abs(outputs-labels).detach().cpu())) #EL2N

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #Lenti

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #ResidualBind
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): #QUIQUIURG
        self.model.eval()
        inputs, labels = batch 
        outputs=self.model(inputs)
        if self.has_aleatoric=='no':
            loss_fn = nn.MSELoss() #.to(device)
            loss = loss_fn(outputs, labels)
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
                vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #Lenti
            spearmanr_vals=np.array(vals)
            #
            vals = []
            for output_index in range(y_score.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #Lenti
            pearsonr_vals=np.array(vals)

            vals = []
            for output_index in range(y_score.shape[-1]):
                vals.append(wpcc_utils.weighted_mse(y_true[:,output_index], y_score[:,output_index], m=-20,M=20, no_weights=True))   # min(data_y)=-6.0841165 np.mean(data_y)=2.1911403e-07 max(data_y)=5.501902
            mse_vals=np.array(vals)

            metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals,'MSE':mse_vals} #,'WPCC':wpcc_vals,'WMSE':wmse_vals}
        else:
            yscore=y_score[0] #QUIQUIURG
            vals = []
            if (yscore==math.nan).all():
                yscore=np.ones(y_score.shape)
            for output_index in range(yscore.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,output_index], yscore[:,output_index])[0] )        
            pearsonr_vals=np.array(vals)
            metrics={'PCC':pearsonr_vals}
        return metrics

    def get_el2n_scores(self): #EL2N
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

    if wanted_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger # https://docs.wandb.ai/guides/integrations/lightning
        wandb_logger = WandbLogger(log_model="all")

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=} {torch.cuda.is_available()=}")

    currdir=os.popen('pwd').read().replace("\n","") #os.getcwd()
    log_dir=outdir+"lightning_logs_"+chosen_model+"/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if wanted_wandb:
        logger_of_choice=wandb_logger
    else:
        logger_of_choice=tb_logger

    model=eval("PL_"+chosen_model+"(input_h5_file='"+inputdir+chosen_dataset+".h5', initial_ds=True, extra_str='"+extra_str+"')") #PERFECT OLD

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
        y_score=model.predict_custom(model.X_test)
        metrics_pretrain=model.metrics(y_score, model.y_test)
    
    if mcdropout_test:
        n_mc = 5
        preds_mc=torch.zeros((n_mc,len(model.X_test)))
        for i in range(n_mc):
            preds_mc[i] = model.predict_custom_mcdropout(model.X_test,
                                                            seed=41+i).squeeze(axis=1).unsqueeze(axis=0)
        metrics_pretrain=model.metrics(y_score, model.y_test)

    trainer = pl.Trainer(accelerator='cuda', devices=-1, max_epochs=model.train_max_epochs, logger=logger_of_choice, callbacks=[callback_ckpt,early_stop_callback],deterministic=True) #, val_check_interval=1) #callbacks=["ADD CALLBACKS", callback_model]) #, logger=None) #,patience=patience) # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) #GOODOLD
    os.system('mv '+inputdir+ckptfile+'-v1.ckpt '+inputdir+ckptfile+'.ckpt')
    if verbose: os.system('date')
    y_score=model.predict_custom(model.X_test)
    if verbose: os.system('date')
    metrics=model.metrics(y_score, model.y_test)

    print(metrics)

    print(ckptfile)
    return metrics










