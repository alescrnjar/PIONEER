import sys
sys.path.append('./src')
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
##import evoaug_AC
##import evoaug
##from evoaug import evoaug, augment
##from evoaug_analysis import utils
#from scipy import stats
import argparse
import tqdm
from FUNCTIONS_4_DALdna import * #plot_distrib

##from tensorboardX import SummaryWriter  
import random
import matplotlib.pyplot as plt

import h5py
import time #QUIQUINONURG temporary

#from torch.multiprocessing import Pool
import torch.multiprocessing as mp

sys.path.append('./blackbox_splice_segmenter/segmenters')
##from resnet import ResNet
##from utils import categorical_crossentropy_2d, spliceAI_data_set, train_model
#from utils_splice_AC import categorical_crossentropy_2d, spliceAI_data_set_AC, train_model, test_model, Ensemble

#from InHouseCNN import *

from filelock import FileLock 

#from Functions_JJD import *
#import Functions_JJD

from pool_gen import SequenceProposer

import bmdal_dholzmueller

import set_torch_tensors_test

parser = argparse.ArgumentParser()

# python DAL_Pipeline_Occasio.py --train_max_epochs 40 --AL_cycles 96 --rank_method mc_dropout_5 --jobseed 1

# Chosen Model
#parser.add_argument('--chosen_model', default='DeepSTARR', type=str)
#parser.add_argument('--chosen_model', default='SpliceAI_full', type=str)
parser.add_argument('--chosen_model', default='InHouseCNN', type=str)
parser.add_argument('--chosen_dataset', default='ATF2_200', type=str)
parser.add_argument('--nickname', default='R', type=str)
parser.add_argument('--chosen_oracle', default='PL_Oracle', type=str)
#parser.add_argument('--initial_dataset', default='default', type=str)
parser.add_argument('--oracle_flag', default='same', type=str)

# Input settings
parser.add_argument('--outdir', default='../outputs_DALdna/', type=str) #'.'
parser.add_argument('--inpdir', default='./inputs/', type=str) 
parser.add_argument('--initial_i_AL', default=0, type=int) 

# Model settings
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--batch_size', default=256, type=int) # orig: 100
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--train_max_epochs', default=3, type=int) # 2 # orig: 100 # Good One for DeepSTARR: 1000
##parser.add_argument('--task_type', default='single_task_binary_classification', type=str)
parser.add_argument('--model_index', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)

parser.add_argument('--patience', default=10, type=int) # orig: 5 #DSRR ONLY
parser.add_argument('--min_delta', default=0.001, type=float)
parser.add_argument('--lr_patience', default=20, type=int) # orig: 2 #QUIQUINONURG this should be added as a parameter for Access_Model
parser.add_argument('--weight_decay', default=1e-6, type=float) #QUIQUINONURG this should be added as a parameter for Access_Model
parser.add_argument('--decay_factor', default=0.1, type=float) #QUIQUINONURG this should be added as a parameter for Access_Model

# Output settings
parser.add_argument('--save_freq', default=1, type=int, help='Frequency for AL cycles of metrics outputting.') #10
parser.add_argument('--sigmadistr_freq', default=10, type=int, help='Frequency of plotting uncertainty distribution') #10

# AL and Seq.Gen. settings
parser.add_argument('--seq_method', default='Xy-from-ds', type=str, help="Method for proposed sequences generation. (options: Xy-from-ds: extract from dataset - random-0.15: 15 percent random mutagenesis - random-0.30: 30 percent random mutagenesis)") 
parser.add_argument('--AL_cycles', default=2, type=int, help="Number of D.A.L. cycles to perform") # 2 5 60 100
parser.add_argument('--N_Active_Learning', default=1, type=int, help="Number of repetitions of D.A.L (e.g. to get errorbars over increase of PCC)")
##parser.add_argument('--initial_fraction', default=0.1, type=float)
parser.add_argument('--incremental_training', default='retrain',type=str,help="0: No incremental training, 1: Incremental training will be performed.") #0 or 1
parser.add_argument('--multiprocesses', default=0,type=int,help="0: No multiprocess training, 1: multiprocess training will be performed.") #0 or 1
parser.add_argument('--how_many_new_batches_at_once', default=1,type=int,help='How many new batches to append at each D.A.L cycle')
#parser.add_argument('--generated_U',default=2560,type=int,help='The number of samples to make the new unlabelled U from which to extract the best batch.') # 1000 10000 dsrr:100000 #pre 31 mar 2024
parser.add_argument('--generated_U',default='2560',type=str,help='The number of samples to make the new unlabelled U from which to extract the best batch.') # 1000 10000 dsrr:100000  #ANCHORED
parser.add_argument('--pristine_N',default=2707,type=int, help="Number of sequences to start the training with") #0 100 2000 10000 100000 # 20000 # 600 #dsrr:20000
parser.add_argument('--firstpristine',default=2707,type=int, help="Number of sequences to start the training with (upfront any D.A.L. cycle)") #0 100 2000 10000 100000 # 20000 # 600 #dsrr:20000
##parser.add_argument('--short_test', default=0, type=int) #0 1000
parser.add_argument('--N_Models', default=1, type=int, help="Number of models that will be trained (e.g. useful for Deep Ensemble uncertainty evaluation)") #"Number of repetitions for uncertainty evaluations (e.g. number of models within Deep Ensemble)") #5
#parser.add_argument('--uncertainty_method', default='entropy', type=str, help="Choice of method for uncertainty evaluation (options: - Deep_Ensemble - MC_dropout)") 
#
##parser.add_argument('--rank_method', default='random', type=str) #uncertainty random
parser.add_argument('--uncertainty_method', default='no', type=str, help="Choice of method for uncertainty evaluation (options: - Deep_Ensemble - MC_dropout)") 
parser.add_argument('--uncertainty_weight', default=0.0, type=float)
parser.add_argument('--diversity_method', default='no', type=str) #QUIQUIURG Using this parameter to indicate that we should use sampeling instead of topK in without removing other functions TODO: add a seperate parameter instead. 
parser.add_argument('--diversity_weight', default=0.0, type=float) #QUIQUIURG Using this parameter to indicate Beta without removing other funtions TODO: add a seperate parameter instead.
parser.add_argument('--highpred_method', default='no',type=str)
parser.add_argument('--highpred_weight', default=0.0, type=float)
parser.add_argument('--sp_desideratum', default='no', type=str)
parser.add_argument('--selection_first', default='no', type=str)
parser.add_argument('--mutrate', default=0.25, type=float)
#
##parser.add_argument('--pristine_method', default='random', type=str, help="Method for generation of initial sequences (options: - random - FF-in-AE : farthest first trasversal in autoencoder latent space)") 
parser.add_argument('--pristine_method', default='ds', type=str, help="Should the pristine labels come from dataset (ds) or from oracle?")
parser.add_argument('--jobseed', default=0, type=int, help="Random seed associated with multitask job scheduler.")
parser.add_argument('--perc_stop', default=0.0, type=float, help="Percentage of the Oracle performance to reach for calling an early stopping of the D.A.L. pipeline (0.0: no early stopping)")
parser.add_argument('--patience_AL', default=5, type=float, help="Number of AL cycles after which to call an early stopping of the D.A.L. pipeline controlled through perc_stop (if perc_stop==0.0: no early stopping)")
parser.add_argument('--what_part', default='one', type=str, help='What DAL cycle part of execute: one, two, or all')

###########################################

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def lock_till_found(file): #QUIQUINONURG this will have to be re-tested, and if not necessary after the insertion of CUDA_VISIBLE_DEVICES, would be better removed.
    ls=os.popen('ls '+file).read()
    while ls=='':
        time.sleep(10)
        ls=os.popen('ls '+file).read()
    return

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #pl.utilities.seed.seed_everything(seed=myseed) # https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything

overall_seed=41 #QUIQUI
set_random_seed(overall_seed)

import math
def tensor_isnan(tensor):
    any=False
    for entry in tensor:
        if math.isnan(entry): any=True
    return any

def force_diversity(new_batch_indexes,how_many_batches,batch_size,preds,preds_for_checking): #QUIQUINONURG remove preds_for_checking
    #print("Into diversity.")
    if type(preds)!=np.ndarray:
        preds_cpu=preds.detach().cpu().numpy()
    else:
        preds_cpu=preds
    preds_sorted=np.array(preds_cpu[new_batch_indexes])
    if args.task_type=='single_task_binary_classification':
        arr1full=np.array(new_batch_indexes)[preds_sorted<=0.5]
        arr2full=np.array(new_batch_indexes)[preds_sorted>0.5] # if '=' goes to '<' because it depends on how round operates: https://pytorch.org/docs/stable/generated/torch.round.html
        arr1=(arr1full)[:how_many_batches*int(batch_size/2)]
        arr2=(arr2full)[:how_many_batches*int(batch_size/2)]
        print(f"Length of two arrays: {len(arr1)}, {len(arr2)}, sum: {len(arr1)+len(arr2)}")
        new_batch_indexes_batched=np.concatenate((arr1,arr2)) #QUIQUINONURG what if batchsize odd? and what if you ran out of 1 class?
    #print("Average class for this batch: ",np.mean(preds_cpu[new_batch_indexes_batched]))
    #print("->",preds_cpu[new_batch_indexes_batched])
    #print("->",preds_cpu[arr1[:10]],preds_cpu[arr2[:10]])
    #print()
    #rint(f"--->{preds[arr1].mean()=} {preds[arr2].mean()=} | {abs(0.5-preds[arr1].mean())=} {abs(0.5-preds[arr2].mean())=} ") #{preds[arr1].mean()+preds[arr2].mean()=}") # {preds_for_checking.mean()=}\n")
    #print(f"--->{preds[arr1].round().mean()=} {preds[arr2].round().mean()=}") # {preds_for_checking.mean()=}\n")
    #print(f"--->{preds_for_checking[arr1].mean()=} {preds_for_checking[arr2].mean()=}") # {preds_for_checking.mean()=}\n")
    #print()
    return new_batch_indexes_batched

import kmer_freq_KL
##def kl_kmer(x,p, min_k, max_k):
##    return kmer_freq_KL.KL_divergence(p, np.array(list(kmer_freq_KL.kmer_frequencies_AC(x,min_k=min_k,max_k=max_k).values())))

class Single_Object_Score(): #QUIQUIURG do I need "super"?
    def __init__(self,method_name,score_value):
        self.method_name=method_name
        self.score_value=score_value

import differentiables_AC
from ranker import Ranker, PowerRanker, SoftmaxRanker, SoftrankRanker

def make_h5_data_module(data_X_train,data_y_train,
                     data_X_test,data_y_test,
                     data_X_test2,data_y_test2,
                     data_X_valid,data_y_valid,
                     batch_size=1,
                     flag='_',
                     outdir='./',
                     removal=True): #QUIQUIURG this is a very inefficent way of doing it: can I avoid saving and reloading a h5 file?
    data_module_file=outdir+'dal_dataset_'+flag+'.h5' #QUIQUINONURG not "args"
    print("Making data module file:",data_module_file)
    if removal: os.system('rm '+data_module_file) #QUIQUIURG had to remove this otherwise reading in PART II becomes a problem, but was it necessary instead?
    with FileLock(os.path.expanduser(f"{data_module_file}.lock")):
        with h5py.File(data_module_file, 'w') as f:
            dset = f.create_dataset("X_train", data=data_X_train, compression="gzip")
            dset = f.create_dataset("Y_train", data=data_y_train, compression="gzip")
            dset = f.create_dataset("X_test", data=data_X_test, compression="gzip") 
            dset = f.create_dataset("Y_test", data=data_y_test, compression="gzip") 
            dset = f.create_dataset("X_test2", data=data_X_test2, compression="gzip") 
            dset = f.create_dataset("Y_test2", data=data_y_test2, compression="gzip") 
            dset = f.create_dataset("X_valid", data=data_X_valid, compression="gzip") 
            dset = f.create_dataset("Y_valid", data=data_y_valid, compression="gzip") 
            f.close() #QUIQUINONURG is this needed?
    lock_till_found(data_module_file)
    return h5py.File(data_module_file,'r'), data_module_file

def make_random_indexes_norepetitions(final_len,N_orig_train,already=[],save=True):
    """ Make random indexes, with no repetitions with previously occurred indexes ('already')"""
    inds=list(np.arange(N_orig_train))
    for alr in already:
        if alr in inds:
            inds.remove(alr)
    inds=np.array(inds)
    np.random.shuffle(inds)
    inds=inds[:final_len]
    new_already=already
    if save:
        for ind in inds:
            new_already.append(ind)
    return inds,new_already

###############

from PL_Models import *
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def Select_Model(chosen_model, 
                 batch_size=None,train_max_epochs=None,patience=None,min_delta=None,
                 #data_module=None,
                 input_h5_file=None,
                 lr=None,
                 initial_ds=False,
                 special_setting=None,
                 ):
    strings_list=[]
    if batch_size!=None: strings_list.append('batch_size='+str(batch_size))
    if train_max_epochs!=None: strings_list.append('train_max_epochs='+str(train_max_epochs))
    if patience!=None: strings_list.append('patience='+str(patience))
    if min_delta!=None: strings_list.append('min_delta='+str(min_delta))
    if input_h5_file!=None: strings_list.append('input_h5_file="'+str(input_h5_file)+'"')
    #if data_module!=None: strings_list.append('data_module=data_module')
    if lr!=None: strings_list.append('lr='+str(lr))
    if initial_ds!=None: strings_list.append('initial_ds='+str(initial_ds))
    if special_setting!=None: strings_list.append('extra_str="'+str(special_setting)+'"')
    strings=''
    for i_str,string in enumerate(strings_list):
        strings+=string
        if i_str!=len(strings_list)-1: strings+=', '
    Model=eval('PL_'+chosen_model+'('+strings+')')
    return Model

def Access_Model(chosen_model,
                 #data_module,
                 #input_h5_file,
                 Train,
                 model_file,
                 outdir,
                 
                 batch_size=None,train_max_epochs=None,patience=None,min_delta=None,
                 #data_module=None,
                 input_h5_file=None,
                 lr=None,
                 initial_ds=False,
                 flag='_',
                 secondtest=False,
                 special_setting=''
                 ): ###PL
                 #hyperparams):
    print(f"QQQQ Access_Model {input_h5_file=}")
    model_file1=model_file
    if ".ckpt" not in model_file1: model_file1+=".ckpt"
    #""
    #if input_h5_file==None: 
    #    Model=Select_Model(chosen_model=chosen_model)
    #else:
    #    Model=Select_Model(chosen_model=chosen_model,input_h5_file=input_h5_file)
    if special_setting=='':
        Model=Select_Model(chosen_model=chosen_model, batch_size=batch_size, train_max_epochs=train_max_epochs, patience=patience, min_delta=min_delta, input_h5_file=input_h5_file, lr=lr, initial_ds=initial_ds)
    else:
        Model=Select_Model(chosen_model=chosen_model, batch_size=batch_size, train_max_epochs=train_max_epochs, patience=patience, min_delta=min_delta, input_h5_file=input_h5_file, lr=lr, initial_ds=initial_ds, special_setting=special_setting)
    #""
    print(f"\n\n----------->AM: {input_h5_file=} {Model.input_h5_file=}\n\n")
    """
    if data_module==None: 
        Model=Select_Model(chosen_model=chosen_model)
    else:
        Model=Select_Model(chosen_model=chosen_model,data_module=data_module)
    """
    parentdirlist=outdir.split('/')[:-2]
    parentdir='/'.join(parentdirlist)+'/'
    parentdirlist1=outdir.split('/')[:-3]
    inputdir=os.getcwd()+'/inputs/'
    #tb_logger = pl_loggers.TensorBoardLogger(save_dir=parentdir+"lightning_logs_"+Model.name+"/")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=parentdir+"lightning_logs_"+flag+"/")

    if 'DAL' in model_file1: #QUIQUINONURG This is too crude, find a better way
        savedir=outdir
    else:
        savedir=inputdir

    Eval=True
    print("Loading dataloaders...")
    time_i=time.time()
    if Train: #QUIQUINONURG the loader should only be evoked once, outside Chosen_Model: otherwise it has to be done 5 times, for Deep Ensembles
        train_dataloader=torch.utils.data.DataLoader(list(zip(Model.X_train,Model.y_train)), batch_size=Model.batch_size, shuffle=True) #.to(device) #QUIQUIURG are there more shuffle=False to remove?
        valid_dataloader=torch.utils.data.DataLoader(list(zip(Model.X_valid,Model.y_valid)), batch_size=Model.batch_size, shuffle=True) #.to(device)
    if Eval:
        test_dataloader=torch.utils.data.DataLoader(list(zip(Model.X_test,Model.y_test)), batch_size=Model.batch_size, shuffle=True) #.to(device) #QUIQUINONURG has this line become redundant?
        #test_dataloader_X=torch.utils.data.DataLoader(list(Model.X_test), batch_size=1, shuffle=True) #.to(device) #batch_size=1 here is necessary, otherwise the data will be organized in lists of batch_size length, not good for calculating y_score. QUIQUINONURG BUT: it may be slow!
        if secondtest: test2_dataloader=torch.utils.data.DataLoader(list(zip(Model.X_test2,Model.y_test2)), batch_size=Model.batch_size, shuffle=True) #.to(device) #QUIQUINONURG has this line become redundant?
    time_f=time.time()
    load_time=time_f-time_i
    print(f"Dataloaders loading time: {load_time}")

    if not Train: #QUIQUIURG is "and not Train" right to put? this will have to change for incremental
        #if os.path.isfile(model_file): #SAVE GOODOLD - NO
        #if os.path.isfile(savedir+model_file): #SAVE GOODOLD - NO
        if os.path.isfile(savedir+model_file1): #SAVE GOODOLD - NO
            #Model.load_state_dict(torch.load(model_file))
            #Model=Model.load_from_checkpoint(model_file) #SAVE GOODOLD - NO
            print(f"----------->AM just pred load: {Model.input_h5_file=}")
            #Model=Model.load_from_checkpoint(savedir+model_file1) # First PL with AL_cycles=10 #PRE 17Dec2023
            #Model=Model.load_from_checkpoint(savedir+model_file1, input_h5_file=input_h5_file) #GOODOLD TILL 16 JUNE 2024 # First PL with AL_cycles=10 #QUIQUIURG what else should I specify? extra_str, wanted_initial_attention, ...? do findergrep . py load_from_checkpoint to verify more recent python scripts
            Model=Model.load_from_checkpoint(savedir+model_file1, input_h5_file=input_h5_file, extra_str=special_setting) # First PL with AL_cycles=10 #QUIQUIURG what else should I specify? extra_str, wanted_initial_attention, ...? do findergrep . py load_from_checkpoint to verify more recent python scripts
            #Model=Model.load_from_checkpoint(savedir+model_file1, input_h5_file=input_h5_file) #QUIQUIURG this SHOULD BE the most correct, but in principle it should not change anything because the test set remains the same all the time.
            #Model=PL_InHouseCNN.load_from_checkpoint(model_file) 
            print(f"----------->AM post load: {Model.input_h5_file=}")
            #Model.input_h5_file=input_h5_file #pre18Dec: with input_h5_file=input_h5_file in model.load this line is unnecessary and inconsequential
            print(f"----------->AM post adj: {Model.input_h5_file=}")
        else:
            #print("Error: set to load file "+model_file+" but not found.") #SAVE GOODOLD - NO
            #print("Error: set to load file "+savedir+model_file+" but not found.")
            print("Error: set to load file "+savedir+model_file1+" but not found.")
            exit()

    #callback_ckpt = pl.callbacks.ModelCheckpoint(dirpath=outdir, filename=model_file, save_weights_only=True)
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt]) #, logger=None) #,patience=patience)
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger) #, logger=None) #,patience=patience) #SAVE GOODOLD - NO

    callback_ckpt = pl.callbacks.ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
            #gpus=1, #QUIQUIURG want to modify for multiple gpus? or does it go automatically?
            #auto_select_gpus=True,
            monitor='val_loss', #default is None which saves a checkpoint only for the last epoch.
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=savedir, #get_github_main_directory(reponame='DALdna')+"inputs/", 
            filename=model_file1.replace('.ckpt',''), #comment out to verify that a different epoch is picked in the name.
        )
    early_stop_callback = pl.callbacks.EarlyStopping(
                              monitor='val_loss',
                              min_delta=Model.min_delta, #QUIQUINONURG (but typically 0.17543655633926392). BUT VARIES FOR DIFFERENT MODELS
                              patience=Model.patience,
                              verbose=False,
                              mode='min'
                              )
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt, early_stop_callback]) 
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt, early_stop_callback], deterministic=True) #PERFECT OLD
    trainer = pl.Trainer(accelerator='cuda', devices=-1, max_epochs=Model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt, early_stop_callback], deterministic=True) 
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger, callbacks=[callback_ckpt]) #CALLBACK
    #trainer = pl.Trainer(max_epochs=Model.train_max_epochs, logger=tb_logger)  #SAVE GOODOLD - YES
    
    if Train: 
        trainer.fit(Model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        #torch.save(Model.state_dict(),model_file) 
        #trainer.save_checkpoint(model_file) #https://pytorch-lightning.readthedocs.io/en/0.8.5/weights_loading.html #SAVE GOODOLD - NO
        #trainer.save_checkpoint(savedir+model_file+'.ckpt') #https://pytorch-lightning.readthedocs.io/en/0.8.5/weights_loading.html #SAVE GOODOLD - YES #CALLBACK : remove for callback
        print(f"CKPT SAVE CHECK: {savedir=} {model_file=}")
        if os.path.isfile(savedir+model_file+"-v1.ckpt"):
            os.system('mv '+savedir+model_file+'-v1.ckpt '+savedir+model_file+'.ckpt')
        """
        np.save(outdir+'el2n-per-epoch.npy',Model.get_el2n_scores()) #EL2N
        # Last error for PL_mpra: numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1
        """

    trainer.test(Model, dataloaders=test_dataloader)
    #y_score=torch.tensor(trainer.predict(Model, dataloaders=test_dataloader_X)) #trainer.predict(...)=[tensor([[2.2167e-06],[1.0000e+00],[9.9964e-01],...,[8.8412e-05]])] #QUIQUINONURG for some reason this was leading to acc=0.5. Why???? 
    print(f"QQQQ {Model.input_h5_file=}")
    print(f"QQQQ {input_h5_file=}")
    print(f"QQQQ {Train=}")
    print(f"QQQQ {Model.X_train[0].shape=}")
    print(f"QQQQ {Model.X_test[0].shape=}")
    y_score=Model.predict_custom(Model.X_test)
    metrics_test=Model.metrics(y_score = y_score, y_true=Model.y_test) #QUIQUINONURG this will have to substitute infere_orcle, paying attention at the rounding/not rounding when necessary.
    #print(metrics)
    #exit()

    if secondtest:
        trainer.test(Model, dataloaders=test2_dataloader) #QUIQUINONURG is this line necessary?
        #y_score=torch.tensor(trainer.predict(Model, dataloaders=test_dataloader_X)) #trainer.predict(...)=[tensor([[2.2167e-06],[1.0000e+00],[9.9964e-01],...,[8.8412e-05]])] #QUIQUINONURG for some reason this was leading to acc=0.5. Why???? 
        y_score2=Model.predict_custom(Model.X_test2)
        metrics_test2=Model.metrics(y_score = y_score2, y_true=Model.y_test2) #QUIQUINONURG this will have to substitute infere_orcle, paying attention at the rounding/not rounding when necessary.
        metrics={}
        for key in metrics_test.keys():
            metrics[key]=metrics_test[key]
        for key in metrics_test2.keys():
            metrics[key+'2']=metrics_test2[key]
    else:
        metrics=metrics_test

    return Model,metrics 

def Access_Model_evoaug(chosen_model,
                 #data_module,
                 #input_h5_file,
                 Train,
                 model_file,
                 outdir,
                 
                 batch_size=None,train_max_epochs=None,patience=None,min_delta=None,
                 #data_module=None,
                 input_h5_file=None,
                 lr=None,
                 initial_ds=False,
                 flag='_',
                 secondtest=False,
                 special_setting=''
                 ): ###PL
                 #hyperparams):
    return Model,metrics 









###################################################################################################################


class Deep_Active_Learning_Cycles:
    def __init__(self, initial_i_AL, model_first_index,
            chosen_model, #task_type, 
            Oracle_Model, data_module, orig_data_module, data_file, #X_valid, y_valid, 
            outflag, nickname,
            seed_add, outdir, #QUIQUIURG X_valid should be part of data_module anyway, no? Or maybe not for DSRR?
            batch_size, already, save_freq,
            AL_cycles, N_Models, train_max_epochs, learning_rate, patience, min_delta,
            incremental_training, sigmadistr_freq,
            seq_method, 
            #rank_method, 
            uncertainty_method,uncertainty_weight,
            diversity_method,diversity_weight,
            highpred_method,highpred_weight,
            sp_desideratum, mutrate,
            generated_U, how_many_new_batches_at_once):
        # Generic attributes:
        self.current_i_AL = initial_i_AL
        self.chosen_model = chosen_model

        if seq_method!='saliency_aleat': #  or 'two_' in args.chosen_dataset: #QUIQUIURG or ok?
            dummy_model=Select_Model(chosen_model=chosen_model, initial_ds=True) #, input_h5_file='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5', initial_ds=initial_ds) #QUIQUINONURG defining a model for nothing isnt great #QUIQUINONURG not args.
        else:
            dummy_model=Select_Model(chosen_model=chosen_model, initial_ds=True,special_setting='heteroscedastic')
        self.task_type=dummy_model.task_type 
        self.metric_names=dummy_model.metric_names
        ##self.task_type= task_type

        self.model_first_index = model_first_index
        self.Oracle_Model = Oracle_Model
        self.data_module = data_module #initially it will be just the pristine
        self.h5_data_module_file = data_file #outdir+'dal_dataset_pristine_seed-'+str(args.jobseed)+'-'+str(model_first_index)+'.h5' # QUIQUINONURG not args
        self.orig_data_module = orig_data_module
        self.batch_size = batch_size
        self.already = already # indexes already included in the dataset                        ## SAVE FOR BASH
        self.save_freq = save_freq
        #self.X_valid = X_valid #QUIQUISOLVED? obsolete!!!
        #self.y_valid = y_valid
        self.seed_add = seed_add

        self.updated_full_X=np.array(data_module['X_train'])
        self.updated_full_y=np.array(data_module['Y_train'])
        self.N_orig_train=len(orig_data_module['X_train']) #QUIQUISOLVED? it is not identical to how it is defined in the main but should be ok?

        self.outflag=outflag
        self.outdir=outdir
        self.nickname=nickname

        self.DAL_Models=[]

        # Part I attributes:
        self.AL_cycles = AL_cycles
        self.N_Models = N_Models
        self.train_max_epochs = train_max_epochs 
        self.incremental_training = incremental_training
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta

        # Part II attributes:
        self.seq_method = seq_method
        if seq_method=='Xy-from-ds': #QUIQUINONURG should be ok but double check
            self.secondtest=False
        else:
            self.secondtest=True
        #self.rank_method = rank_method
        self.uncertainty_method=uncertainty_method
        self.uncertainty_weight=uncertainty_weight
        self.diversity_method=diversity_method
        self.diversity_weight=diversity_weight
        self.highpred_method=highpred_method
        self.highpred_weight=highpred_weight
        #self.generated_U = generated_U #pre 31 mar 2024
        if 'notanch-' in generated_U:
            self.anchored=False
            self.screm=False
            self.generated_U=int(generated_U.replace('notanch-',''))
        elif 'screm-' in generated_U:
            self.anchored=True
            self.screm=True
            self.generated_U=int(generated_U.replace('screm-',''))
        else:
            self.anchored=True
            self.screm=False
            self.generated_U=int(generated_U)
        self.rank_method=uncertainty_method+'_'+str(uncertainty_weight)+'_'+diversity_method+'_'+str(diversity_weight)+'_'+highpred_method+'_'+str(highpred_weight)
        self.sp_desideratum=sp_desideratum

        if self.seq_method=='saliency_aleat':
            self.special_setting='heteroscedastic'
        else:
            self.special_setting=''

        self.mutrate=mutrate

        self.sigmadistr_freq = sigmadistr_freq
        self.how_many_new_batches_at_once = how_many_new_batches_at_once

        # Metrics
        self.monitored_part1={}
        #self.monitored_part1=self.monitored_part1 | {key:[] for key in self.metric_names} #pre 25 mar 2024
        #self.monitored_part1=self.monitored_part1 | {key+'2':[] for key in self.metric_names} #pre 25 mar 2024
        for key in self.metric_names:
            self.monitored_part1[key]=[]
            self.monitored_part1[key+'2']=[]

        self.monitored_part2={}
        #self.monitored_part2=self.monitored_part2 | {'cum_perc_uncs':[], 'unc_thrs':[]} #pre 25 mar 2024
        self.monitored_part2['cum_perc_uncs']=[]
        self.monitored_part2['unc_thrs']=[]
        if 'classif' in self.task_type:
            #self.monitored_part2=self.monitored_part2 | {'averclass':[]} #pre 25 mar 2024 #QUIQUINONURG remove or port it to PL_Models.py: maybe it's not so necessary, once it has been verified?
            self.monitored_part2['averclass']=[]
        #if N_Models>1 and uncertainty_method=='sigma_deep_ensemble': 
        if N_Models>1 and sp_desideratum=='sigma_deep_ensemble': 
            #self.monitored_part2=self.monitored_part2 | {'Cumulative-'+key:[] for key in self.metric_names}  #pre 25 mar 2024
            #self.monitored_part2=self.monitored_part2 | {'Cumulative-'+key+'2':[] for key in self.metric_names}  #pre 25 mar 2024
            for key in self.metric_names:
                self.monitored_part2['Cumulative-'+key]=[]
                self.monitored_part2['Cumulative-'+key+'2']=[]
                                    

        """
        if task_type=='single_task_regression':
            self.monitored_part1={'PCC':[],'Spearman':[]} #QUIQUINONURG should eventually become pearsonr and spearmanr
            self.monitored_part2={'cum_perc_uncs':[],'unc_thrs':[]}
        elif 'classif' in self.task_type:
            #self.monitored_part1={'accuracy':[],'AUROC':[],'AUPR':[]}
            self.monitored_part1={'accuracy':[],'AUROC':[],'AUPR':[], 'accuracy2':[],'AUROC2':[],'AUPR2':[], 'accuracytrain':[],'AUROCtrain':[],'AUPRtrain':[]} #QUIQUIURG we may want the metrics to be derived from the Model
            self.monitored_part2={'cum_perc_uncs':[],'unc_thrs':[],'averclass':[]} #,'CumAcc':[]}
            if N_Models>1 and uncertainty_method=='sigma_deep_ensemble': #CUMACC
                if self.task_type=='single_task_regression':
                    self.monitored_part2['CumPCC']=[] 
                    self.monitored_part2['CumSpearman']=[] 
                    self.monitored_part2['CumPCC2']=[] 
                    self.monitored_part2['CumSpearman2']=[] 
                elif 'classif' in self.task_type:
                    self.monitored_part2['CumAcc']=[] #QUIQUISOLVED eventually will need CumAverclass too. NO! Bc it's the singlely-produced dataset in part2!
                    self.monitored_part2['CumAUPR']=[] 
                    self.monitored_part2['CumAUROC']=[] 
                    self.monitored_part2['CumAcc2']=[] #QUIQUISOLVED eventually will need CumAverclass too. NO! Bc it's the singlely-produced dataset in part2!
                    self.monitored_part2['CumAUPR2']=[] 
                    self.monitored_part2['CumAUROC2']=[] #QUIQUIURG by making it a monitored of part2, cumulative perfomance metrics are ON THE NEW DATASET??? And therefore are more accurate? No! Bc valid and test remain the same.
        """

        self.wanted_wandb=False
        #self.wanted_wandb=True
        if self.wanted_wandb: 
            import wandb
            os.system('wandb login 7693bea9dff37d98f5d98928ab2fbe3842552e84')
            #os.system('export WANDB_API_KEY=$(cat wandb_api_key_acrnjar.txt)') #should go in the job scheduler # https://wandb.ai/acrnjar
            wandb_project = config['wandb_'+chosen_model+'_'+chosen_dataset] # project name
            wandb_name = config['wandb_DAL_'+chosen_model+'_'+chosen_dataset+'_seed-'+str(myseed)] # give a run/trial name here
            wandb.init(project=wandb_project, name=wandb_name, config=config)
                    
    def reset_for_new_AL_cycle(self):
        self.DAL_Models=[]
        #QUIQUIURG there shouldn't be anything to reinitialize every time, right?

    """
    def empty_monitored_part1(self):
        if self.task_type=='single_task_regression':
            empty_monitored={'PCC':[],'Spearman':[]} #,'cum_perc_uncs':[],'unc_thrs':[]}
        elif 'classif' in self.task_type:
            #empty_monitored={'accuracy':[],'AUROC':[],'AUPR':[]} #,'cum_perc_uncs':[],'unc_thrs':[]}
            empty_monitored={'accuracy':[],'AUROC':[],'AUPR':[], 'accuracy2':[],'AUROC2':[],'AUPR2':[], 'accuracytrain':[],'AUROCtrain':[],'AUPRtrain':[]} #,'cum_perc_uncs':[],'unc_thrs':[]}
            if self.N_Models>1 and self.uncertainty_method=='sigma_deep_ensemble': #CUMACC
                if self.task_type=='single_task_regression':
                    empty_monitored['CumPCC']=[] 
                    empty_monitored['CumSpearman']=[] 
                    empty_monitored['CumPCC2']=[] 
                    empty_monitored['CumSpearman2']=[] 
                elif 'classif' in self.task_type:
                    empty_monitored['CumAcc']=[]
                    empty_monitored['CumAUPR']=[]
                    empty_monitored['CumAUROC']=[]
                    empty_monitored['CumAcc2']=[]
                    empty_monitored['CumAUPR2']=[]
                    empty_monitored['CumAUROC2']=[]
        return empty_monitored
    """
    def empty_monitored_part1(self):
        empty_monitored={}
        #empty_monitored=empty_monitored | {key:[] for key in self.metric_names} #pre 25 mar 2024
        #empty_monitored=empty_monitored | {key+'2':[] for key in self.metric_names} #pre 25 mar 2024 
        for key in self.metric_names:
            empty_monitored[key]=[]
            empty_monitored[key+'2']=[]
        return empty_monitored

    def jModel(self,j,local_seed_add,Train=True, special_setting=''):
        print(f"QQQQjModel {self.h5_data_module_file=}")
        return Access_Model(self.chosen_model, 
                                Train=Train,
                                #model_file=self.outdir+"DAL_Model_j-"+str(j)+"_"+self.outflag+"_seedadd-"+str(local_seed_add), #SAVE GOODOLD - NO
                                #model_file="DAL_Model_j-"+str(j)+"_"+self.outflag+"_seedadd-"+str(local_seed_add), #PERFECT OLD
                                model_file="DAL_Model_j-"+str(j)+"_"+self.nickname+"_seedadd-"+str(local_seed_add), #PERFECT OLD
                                #model_file="DAL_Model_j-"+str(j)+"_"+self.outflag+"_seedadd-"+str(local_seed_add)+".ckpt",
                                outdir=self.outdir,
                                
                                batch_size=self.batch_size,train_max_epochs=self.train_max_epochs,patience=self.patience,min_delta=self.min_delta,
                                #data_module=None,
                                ##data_module=self.data_module,
                                ##input_h5_file=self.outdir+'dal_dataset_'+str(self.seed_add)+'.h5'
                                input_h5_file=self.h5_data_module_file,
                                lr=self.learning_rate,
                                initial_ds=False,
                                #flag=self.outflag+'_seed'+str(self.seed_add), #only useful for lightning logs
                                flag=self.nickname+'_seed'+str(self.seed_add), #only useful for lightning logs
                                secondtest=self.secondtest,
                                special_setting=special_setting
                                )

    def Active_Learning_Cycle_Part_1(self):
        """ First Part: Train model (with new batch of data)"""
        set_random_seed(self.seed_add) # QUIQUIURG this is needed in case we do not evoke the Active_Learning_Loop function, as it is set at its beginning. But is it ok to set it here too?

        if self.current_i_AL>0:
            for key in self.monitored_part1.keys(): 
                #self.monitored_part1[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy'))
                self.monitored_part1[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'))

        if self.incremental_training=='retrain': self.reset_for_new_AL_cycle()
        ##if self.uncertainty_method=='Deep_Ensemble': #QUIQUINONURG change name: deep_ens -> number_of_models
        temp_metrics=self.empty_monitored_part1()
        if self.incremental_training=='retrain':

            if args.multiprocesses==0:
                """
                if self.rank_method=='random': #QUIQUIURG
                    jmax=1 # we only want 1 model for training, regardless of how many models we mandated from above
                else:
                    jmax=self.N_Models #QUIQUINONURG this whole part does not require a for loop anymore, given the bash pipeline
                """
                jmax=self.N_Models
                os.system('date')
                timetrain_i=time.time()
                for j in tqdm.tqdm(range(self.model_first_index,self.model_first_index+jmax), total=jmax, desc='Training every model in Deep Ensemble', colour='yellow'):
                    #if self.chosen_model=='SpliceAI_full': print("INTO J LOOP")
                    DAL_Model,metrics=self.jModel(j,self.seed_add,Train=True, special_setting=self.special_setting)
                    #set_random_seed(self.seed_add) #QUIQUIURG re-setting it here create problem for the portability?
                    self.DAL_Models.append(DAL_Model)
                    """
                    if self.task_type=='single_task_regression':
                        temp_metrics['PCC'].append(metrics['PCC'][0]) #p_vals[0])
                        temp_metrics['Spearman'].append(metrics['Spearman'][0]) #p_vals[0])
                    elif 'classif' in self.task_type:
                        for key in temp_metrics.keys():
                            if key!='cum_perc_uncs' and key!='unc_thrs' and key!='averclass': #QUIQUINONURG check if this if is fine
                                if key in list(metrics.keys()): temp_metrics[key].append(metrics[key])
                    """
                    for key in temp_metrics.keys():
                        if key!='cum_perc_uncs' and key!='unc_thrs' and key!='averclass': #QUIQUINONURG check if this if is fine
                            toappend=metrics[key]
                            if '__len__' in dir(toappend):
                                if len(toappend)==2: toappend=toappend[0]
                            if key in list(metrics.keys()): temp_metrics[key].append(toappend)
                print("Trainings in Part 1 completed.")
                os.system('date')
                time1_i=time.time()
                print("Training time:",time1_i-timetrain_i)
            """
            elif args.multiprocesses==1: # 1 as for "True", 0 as for "False" #QUIQUIURG args->self
                with mp.Pool(self.N_Models) as pool:
                    processes=pool.map(self.jModel,range(self.N_Models))
                torch.manual_seed(overall_seed) #QUIQUIURG re-setting it here create problem for the portability?
                random.seed(overall_seed)
                np.random.seed(overall_seed)
                if self.task_type=='single_task_regression':
                    temp_metrics['PCC']=processes['PCC'][0]
                elif 'classif' in self.task_type:
                    temp_metrics[key]=metrics[key]
            """

        elif incremental_train_InHouseCNN=='random_minibatches': #WORKINPROGRESS #https://arxiv.org/pdf/1707.05928.pdf
            # check incremental_train_InHouseCNN() in CNNFrAmb.py
            exit()
        else:
            print("Wrongly selected incremental training mode.")
            exit()

        # Average metrics over the j models       
        """
        if self.task_type=='single_task_regression':
            self.monitored_part1['PCC'].append(np.mean(temp_metrics['PCC']))
            self.monitored_part1['Spearman'].append(np.mean(temp_metrics['Spearman']))
            #self.monitored_part1['PCC'].append(np.array([np.mean(temp_metrics['PCC']),np.std(temp_metrics['PCC'])]))  #QUIQUINONURG
        elif 'classif' in self.task_type:
            for key in temp_metrics.keys():
                if key!='cum_perc_uncs' and key!='unc_thrs': #we do not want to average those metrics, otherwise we'd get a nan because we'd be trying to divide by zero when calculating the average (cum_perc_unc is calculated in part II, not here, so here the list would be empty!) #QUIQUINONURG this if should be useless, since cum_perc_unc and unc_thrs are not keys of self.monitored_part1
                    self.monitored_part1[key].append(np.mean(temp_metrics[key])) 
                    #self.monitored_part1[key].append(np.array([np.mean(temp_metrics[key]),np.std(temp_metrics[key])])) #QUIQUINONURG
        """
        for key in temp_metrics.keys():
            if key!='cum_perc_uncs' and key!='unc_thrs': #we do not want to average those metrics, otherwise we'd get a nan because we'd be trying to divide by zero when calculating the average (cum_perc_unc is calculated in part II, not here, so here the list would be empty!) #QUIQUINONURG this if should be useless, since cum_perc_unc and unc_thrs are not keys of self.monitored_part1
                self.monitored_part1[key].append(np.mean(temp_metrics[key])) 
                #self.monitored_part1[key].append(np.array([np.mean(temp_metrics[key]),np.std(temp_metrics[key])])) #QUIQUINONURG
                    
        # Save monitored metrics relative to training
        if self.current_i_AL % self.save_freq == 0: 
            """
            if self.task_type=='single_task_regression':
                np.save(self.outdir+'Res-'+'PCC_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part1['PCC']))
                np.save(self.outdir+'Res-'+'Spearman_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part1['Spearman']))
            elif 'classif' in self.task_type:
                for key in self.monitored_part1.keys(): 
                    np.save(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part1[key]))
            """
            for key in self.monitored_part1.keys(): 
                #np.save(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part1[key]))
                np.save(self.outdir+'Res-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part1[key]))
                    
        #print(f"\n- AL #{self.current_i_AL} PART I COMPLETE: \nLast {str(list(self.monitored_part1.keys())[0])} monitored: {self.monitored_part1[list(self.monitored_part1.keys())[0]][-1]}")
        print(f"\n- AL #{self.current_i_AL} PART I COMPLETE: Last {str(list(self.monitored_part1.keys())[0])} monitored: {self.monitored_part1[list(self.monitored_part1.keys())[0]][-1]}")
        os.system('date')
        time1_f=time.time()
        print("Post training time:",time1_f-time1_i)
        if self.wanted_wandb:
            wandb.log(self.monitored_part1)

    def load_for_Part_2(self):

        print(f"QQQQ load4part2 {self.h5_data_module_file=}")

        #print("CONTROL: 0")
        #if self.current_i_AL!=0: self.already=list(np.load(self.outdir+'already_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy'))
        if self.current_i_AL!=0: self.already=list(np.load(self.outdir+'already_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'))

        if self.current_i_AL>0:
            for key in self.monitored_part2.keys(): 
                #self.monitored_part2[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy'))
                self.monitored_part2[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'))

        #QUIQUINONURG the loading of the models may only be done if no_sequences=False
        """
        if self.rank_method=='random': #QUIQUIURG
            jmax=1 # we only want 1 model for training, regardless of how many models we mandated from above
        else:
            jmax=self.N_Models
        """
        #print("CONTROL: 0a")
        jmax=self.N_Models
        #if self.current_i_AL>0 and self.uncertainty_method=='sigma_deep_ensemble': #CUMACC
        if self.current_i_AL>0 and self.sp_desideratum=='sigma_deep_ensemble': #CUMACC
            """
            if self.task_type=='single_task_regression':
                self.monitored_part2['CumPCC']=list(np.load(self.outdir+'Res-CumPCC_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumSpearman']=list(np.load(self.outdir+'Res-CumSpearman_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumPCC2']=list(np.load(self.outdir+'Res-CumPCC2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumSpearman2']=list(np.load(self.outdir+'Res-CumSpearman2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
            elif 'classif' in self.task_type:
                self.monitored_part2['CumAcc']=list(np.load(self.outdir+'Res-CumAcc_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumAUPR']=list(np.load(self.outdir+'Res-CumAUPR_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumAUROC']=list(np.load(self.outdir+'Res-CumAUROC_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumAcc2']=list(np.load(self.outdir+'Res-CumAcc2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumAUPR2']=list(np.load(self.outdir+'Res-CumAUPR2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                self.monitored_part2['CumAUROC2']=list(np.load(self.outdir+'Res-CumAUROC2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
            """
            for key in self.monitored_part2.keys():
                if 'Cumulative' in key:
                    #self.monitored_part2[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy')) 
                    self.monitored_part2[key]=list(np.load(self.outdir+'Res-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy')) 

        """
        if self.N_Models>1 and self.uncertainty_method=='sigma_deep_ensemble': #CUMACC
            cum_digits=torch.empty(0) 
            cum_digits2=torch.empty(0)
            cum_gt=torch.empty(0) 
            cum_gt2=torch.empty(0) 
            if 'classif' in self.task_type:
                cum_acc=torch.empty(0) 
                cum_acc2=torch.empty(0) 
        """          
        #print("CONTROL: 0b")     
        #if self.N_Models>1 and self.uncertainty_method=='sigma_deep_ensemble': #CUMACC
        if self.N_Models>1 and self.sp_desideratum=='sigma_deep_ensemble': #CUMACC
            cum_digits=torch.empty(0) 
            cum_digits2=torch.empty(0)
            cum_gt=torch.empty(0) 
            cum_gt2=torch.empty(0) 

        # This loop is not just necessary for the Cumulative metrics
        for jcount,j in tqdm.tqdm(enumerate(range(self.model_first_index,self.model_first_index+jmax)), total=jmax, desc='Loading every model in Deep Ensemble', colour='yellow'):
            DAL_Model,metrics=self.jModel(j,self.seed_add+jcount*1000,Train=False, special_setting=self.special_setting) #QUIQUIURG within load_for_Part_2 I can actually make the average of metrics, to be plotted. #QUIQUIURG is jcount*1000 only a temporary crafting?
            self.DAL_Models.append(DAL_Model)

            print("CONTROL: 0b0")
            #if self.N_Models>1 and self.uncertainty_method=='sigma_deep_ensemble':  #CUMACC
            if self.N_Models>1 and self.sp_desideratum=='sigma_deep_ensemble':  #CUMACC
                digits=torch.tensor(self.DAL_Models[j].predict_custom(torch.tensor(self.data_module['X_test'])))
                digits2=torch.tensor(self.DAL_Models[j].predict_custom(torch.tensor(self.data_module['X_test2'])))
                #print("CONTROL: 0b1")
                gt=torch.tensor(self.data_module['Y_test']) #)).detach().cpu() #.numpy()
                gt2=torch.tensor(self.data_module['Y_test2']) #)).detach().cpu() #.numpy()
                #print("CONTROL: 0b2")
                cum_digits=torch.cat((cum_digits,digits),axis=0)
                cum_digits2=torch.cat((cum_digits2,digits2),axis=0)
                #print("CONTROL: 0b3")
                cum_gt=torch.cat((cum_gt,gt),axis=0)
                cum_gt2=torch.cat((cum_gt2,gt2),axis=0)
                print(f"CONTROLCum: {digits.shape=} {gt.shape=} {digits2.shape=} {gt2.shape=} | {cum_digits.shape=} {cum_gt.shape=} {cum_digits.shape=} {cum_gt.shape=}")
            #print("CONTROL: 0b4")
        
        print("CONTROL: 0c")
        os.system('date')
        #if self.N_Models>1 and self.uncertainty_method=='sigma_deep_ensemble': #CUMACC
        if self.N_Models>1 and self.sp_desideratum=='sigma_deep_ensemble':  #CUMACC
            #cum_metrics=self.DAL_Models[0].metrics(self.DAL_Models[j].predict_custom(torch.tensor(self.data_module['X_test'])), torch.tensor(self.data_module['Y_test'])) #any model of the ensemble will do, it is just to access the metrics function
            cum_metrics=self.DAL_Models[0].metrics(cum_digits, cum_gt) #any model of the ensemble will do, it is just to access the metrics function
            print("CONTROLCum: 0c0")
            os.system('date')
            #cum_metrics2=self.DAL_Models[0].metrics(self.DAL_Models[j].predict_custom(torch.tensor(self.data_module['X_test2'])), torch.tensor(self.data_module['Y_test2'])) #any model of the ensemble will do, it is just to access the metrics function
            cum_metrics2=self.DAL_Models[0].metrics(cum_digits2, cum_gt2)
            print("CONTROLCum: 0c1")
            os.system('date')
            for key in cum_metrics.keys():
                print(f"{self.monitored_part2.keys()=} {cum_metrics.keys()=}")
                self.monitored_part2['Cumulative-'+key].append(cum_metrics[key])
                print("CONTROLCum: 0c2")
                self.monitored_part2['Cumulative-'+key+'2'].append(cum_metrics2[key])
                print("CONTROLCum: 0c3")
                #np.save(self.outdir+'Res-Cumulative-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['Cumulative-'+key]))
                np.save(self.outdir+'Res-Cumulative-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['Cumulative-'+key]))
                print("CONTROLCum: 0c4")
                #np.save(self.outdir+'Res-Cumulative-'+key+'2_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['Cumulative-'+key+'2']))
                np.save(self.outdir+'Res-Cumulative-'+key+'2_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['Cumulative-'+key+'2']))
                print("CONTROLCum: 0c5")
        
        print("CONTROL: 0d")

    def propose_new_sequences(self): #, no_sequences):
        no_sequences=False
        # Propose new sequences
        print(f"SP CONTROL 0 {self.seq_method=}")
        if self.seq_method=='Xy-from-ds':
            print("SP CONTROL 1, Xy")
            length4rand=self.N_orig_train
            randinds,self.already=make_random_indexes_norepetitions(length4rand,self.N_orig_train,already=self.already,save=False) #if you save here, all generated_U will be appended, but we want instead only the new selected batch to also add to update updated_full_X
            randinds=np.sort(randinds)
            if len(randinds)!=0: 
                if len(randinds)!=len(np.unique(randinds)): 
                    print("ERROR! randinds not unique!")
                    exit()
                proposed_X=self.orig_data_module['X_train'][randinds] #QUIQUIURG is randinds originally calculated over orig_data_module? print sovrapp of randinds and self.already? 
                proposed_y=self.orig_data_module['Y_train'][randinds] #QUIQUIURG is randinds originally calculated over orig_data_module? print sovrapp of randinds and self.already?
            else:
                no_sequences=True
        #elif self.seq_method=='XdsYor' or self.seq_method: #QUIQUIURG this used to be for supergroup, why that "or self.seq_method"?
        elif self.seq_method=='XdsYor': 
            print("SP CONTROL 1, Xfrom")
            length4rand=self.N_orig_train
            randinds,self.already=make_random_indexes_norepetitions(length4rand,self.N_orig_train,already=self.already,save=False) #if you save here, all generated_U will be appended, but we want instead only the new selected batch to also add to update updated_full_X
            #print(f"RECOVPOOL: {length4rand=} {self.N_orig_train=} {self.already=}")
            print(f"RECOVPOOL: {length4rand=} {self.N_orig_train=} {len(self.already)=} {len(randinds)=}")
            randinds=np.sort(randinds)
            if len(randinds)!=0: 
                if len(randinds)!=len(np.unique(randinds)): 
                    print("ERROR! randinds not unique!")
                    exit()
                proposed_X=self.orig_data_module['X_train'][randinds] #QUIQUIURG is randinds originally calculated over orig_data_module? print sovrapp of randinds and self.already? 
                ##print(f"ALCMSALCKSA {proposed_X.shape=} {type(proposed_X)=}")
                #if type(proposed_X)==np.array: proposed_X=torch.tensor(proposed_X)
                #proposed_y=self.Oracle_Model.predict_custom(torch.tensor(proposed_X)).detach().cpu().numpy() #goodold #QUIQUIURG(imported from another line) non sono sicuro che questo funzioni, inoltre: e sicuramente lento: va fatto in un data_loader: potresti farlo dentro una function #QUIQUIURG this should not appear at this stage: otherwise the all point of the DAL of not consulting the Oracle too often is lost!
                proposed_y=self.Oracle_Model.interrogate(torch.tensor(proposed_X)).detach().cpu().numpy() #QUIQUIURG(imported from another line) non sono sicuro che questo funzioni, inoltre: e sicuramente lento: va fatto in un data_loader: potresti farlo dentro una function #QUIQUIURG this should not appear at this stage: otherwise the all point of the DAL of not consulting the Oracle too often is lost!
            else:
                print("RECOVPOOL: Xfrom will result in no_seqs True.")
                #print(f"WARNING: {len(randinds)=}")
                #if self.chosen_model=='InHouseCNN':
                #    proposed_X=torch.empty((0,self.orig_data_module['X_train'].shape[1],self.orig_data_module['X_train'].shape[2]))
                #    proposed_y=torch.empty((0,self.orig_data_module['y_train'].shape[1]))
                no_sequences=True
        elif self.seq_method in ['totally_random', 
                                 'mutation', 
                                 'saliency', 'saliency_y', 'saliency_div_y', 'saliency_aleat', 'saliency_evidential', 'saliency_U-A',
                                 'hessian', # AC
                                 'salfirstlayer', #AC
                                 'simulated_annealing', 'simulated_annealing_y',
                                 'greedy', 'greedy_y', 
                                 'genetic','genetic_y',
                                 'evoaug', #AC
                                 'evoaugassign', #AC
                                 'evoaugmut', #AC
                                 'realevoaug', #AC
                                 'totally_random_then_saliency', #AC
                                 'mutation_then_saliency', #AC
                                 'fromfile', #AC
                                 'fromfile_then_saliency', #AC
                                 'dinuc', #AC 
                                 'dinuc_then_saliency', #AC 
                                 'vanilla_diffusion', #AC
                                 'diffusion_file', #AC
                                 'diffusion_y', #AC
                                 'GradientSHAP', #AC
                                 'DeepLiftSHAP', #AC
                                 'motifembed', #AC
                                 'dinuc_then_motifembed', #AC 
                                 'realmut', #AC
                                 'BatchBALD', #AC
                                 'BatchBALDsubsel', #AC
                                 'BADGEsubsel', #AC
                                 'LCMDsubsel', #AC
                                 'concatBADGE1', #AC
                                 'concatLCMD1', #AC
                                 'concatrand1', #AC
                                 'BADGEfromt', #AC
                                 'LCMDfromt', #AC
                                 'LCMDfromd', #AC
                                 'LCMDfromJ', #AC
                                 'realsal', #AC
                                 'Costmixrand1', #AC
                                 'CostmixLCMD1', #AC

                                 'realTEMPAsal', #AC
                                 'CostmixTEMPArand1', #AC
                                 ##'CostmixTEMPALCMD1', #AC This method is not actually a thing
                                 'concatTEMPALCMD1', #AC
                                 'concatTEMPArand1', #AC
                                 'PriceHundredLCMD', #AC
                                 'Price20KLCMD', #AC
                                 ]:
            print("SP CONTROL 1, generation")
            #seq_method0=self.seq_method
            #if self.seq_method=='totally_random_then_saliency':
            if '_then_' in self.seq_method:
                seq_method0=self.seq_method.split('_then_')[0] #'totally_random'
                seq_method1=self.seq_method.split('_then_')[1] #'saliency'
                seq_methods=[seq_method0,seq_method1]
                #sp1 = SequenceProposer(generation_method=seq_method1, 
                #                  sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add, 
                #                  track_time=False, track_uncertanties = False, 
                #                  track_batches=False, track_hamming=False, 
                #                  track_pref = self.outdir+self.outflag)  #QUIQUINONURG should this outflag be changed to nickname?
            else:
                seq_methods=[self.seq_method]

            for i_sm,seq_method_i in enumerate(seq_methods):
                #sp = SequenceProposer(generation_method=self.seq_method, #SEQMETHOD0
                sp = SequenceProposer(generation_method=seq_method_i, #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) #QUIQUINONURG should this outflag be changed to nickname?

                print("SP CONTROL 1, generation 0")
                ranker_sp = Ranker(self.DAL_Models,
                                self.batch_size,
                                how_many_batches=self.how_many_new_batches_at_once,
                                #uncertainty_method=self.uncertainty_method,
                                uncertainty_method=self.sp_desideratum,
                                diversity_method=self.diversity_method, #QUIQUINONURG make distinct for ranker_sp or remove
                                highpred_method=self.highpred_method, #QUIQUINONURG make distinct for ranker_sp or remove
                                uncertainty_weight=1.0, #QUIQUINONURG make distinct for ranker_sp or remove
                                diversity_weight=self.diversity_weight, #QUIQUINONURG make distinct for ranker_sp or remove
                                highpred_weight=self.highpred_weight, #QUIQUINONURG make distinct for ranker_sp or remove
                                chosen_model=self.chosen_model, 
                                cycle=1, sigmadistr_freq=1, device='cuda', #QUIQUI IDK what the right args for these are
                                outdir=self.outdir, outflag=self.outflag,    #QUIQUINONURG should this outflag be changed to nickname?
                                #local_seed=self.seed_add, task_type=self.task_type) #before 15 mar 2024
                                local_seed=self.seed_add+222*self.current_i_AL, task_type=self.task_type)
                print("SP CONTROL 1, generation 1")
                #Setting a bunch of hyper parameters 
                #QUIQUInonurg 
                #TODO: do some sort of optimization over these, or allow them to be set by user...
                #n_to_make = self.batch_size*self.how_many_new_batches_at_once
                n_to_make=self.generated_U
                mut_per_cycle = 1
                #mutant_fraction = 0.25 #goodold pre 24 Jan 2024
                if i_sm==0:
                    mutant_fraction = self.mutrate #QUIQUIURG this only works for totally_random_then_saliency
                else:
                    mutant_fraction = self.mutrate #0.10 # AC: The best
                cycles = int(self.updated_full_X.shape[-1]*mutant_fraction)
                #if cycles==0: #QUIQUIURG this would better be inserted
                #    print("ERROR: cycles results = 0.") 
                #    exit()
                if i_sm==0:
                    if self.anchored==True: #ANCHORED
                        X_source = self.orig_data_module['X_train'] #AC This was double checked with Jack, after a doubt during a meeting with Peter
                        print(f"as orig_data_module: {len(X_source)=}")
                    else:
                        centr_inds=np.random.choice(np.arange(len(self.updated_full_X)),size=len(self.orig_data_module['X_train']),replace=False) #QUIQUIURG is this correct????
                        X_source=self.updated_full_X[centr_inds]
                else:
                    X_source = proposed_X 
                    print(f"as identical to proposed_X: {len(X_source)=}")
                #temp = 0.00001 #PERFECT OLD 13 Aug 2024
                #if self.mutrate==0.25:
                if 'newLentiMPRAK562' in args.chosen_dataset: #QUIQUINONURG remove args
                    if self.mutrate>0.15: # AC: circa 35 / 230. Ok approximated, if you dont use anything between 30/230=0.13 and  40/230=0.1739
                        temp = 0.00001
                    #elif self.mutrate==0.1:
                    else:
                        temp = 0.000001
                elif 'newLentiMPRAHepG2' in args.chosen_dataset: #QUIQUINONURG remove args
                    temp = 0.00001
                elif 'RBFOX1' in args.chosen_dataset: #QUIQUINONURG remove args
                    if self.mutrate>0.19: # AC: circa 8/41. Ok approximated, if you dont use anything between 6/41=0.146 and 8/41=0.195
                        temp = 0.0001
                    else:
                        temp = 0.00001
                elif 'DeepSTARR' in args.chosen_dataset:
                    temp = 0.00001

                if 'TEMPA' in seq_method_i:
                    temp = 'neg_inf'

                #else:
                #    print("ERROR: wrong mutrate for temperature")
                #    exit()
                #temp='neg_inf'
                #if 'hessian' in self.seq_method: temp='neg_inf' #QUIQUIURG this is temporary
                if 'hessian' in seq_method_i: temp='neg_inf' #QUIQUIURG this is temporary
                decay = 0
                x_prob = 0.1
                expansion_fold = 10
                
                print("SP CONTROL 1, generation 2")
                if seq_method_i=='totally_random': #QUIQUINONURG should this be removed? Since it s not correct?
                    proposed_X=sp.generate_batch(n_to_make, ranker=ranker_sp)
                    print(f"First time a proposed_X ever appears (tot rand): {len(proposed_X)=}")
                    
                elif seq_method_i=='mutation':
                    print("SP CONTROL 1, generation 3")
                    #proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                        mut_per_cycle, cycles=cycles, 
                    #                        ranker=ranker_sp) # pre Feb 22 2024
                    proposed_X=sp.generate_batch(n_to_make, X_source, 
                                            mut_per_cycle, cycles=cycles, 
                                            ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    print(f"First time a proposed_X ever appears (mutation): {len(proposed_X)=} {n_to_make=} {mut_per_cycle=} {cycles=}")
                    print("SP CONTROL 1, generation 4")
                elif seq_method_i=='realmut':
                    print("SP CONTROL 1, generation 3")
                    #proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                        mut_per_cycle, cycles=cycles, 
                    #                        ranker=ranker_sp) # pre Feb 22 2024
                    proposed_X=sp.generate_batch(n_to_make, X_source, 
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    print(f"First time a proposed_X ever appears (realmut): {len(proposed_X)=} {n_to_make=} {mut_per_cycle=} {cycles=}")
                    print("SP CONTROL 1, generation 4")
                elif seq_method_i=='BatchBALDsubsel':
                    print("SP CONTROL 1, generation 3")
                    #proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                        mut_per_cycle, cycles=cycles, 
                    #                        ranker=ranker_sp) # pre Feb 22 2024
                    proposed_X=sp.generate_batch(n_to_make, X_source, 
                                            mut_per_cycle, cycles=cycles, 
                                            ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    print(f"First time a proposed_X ever appears (mutation): {len(proposed_X)=} {n_to_make=} {mut_per_cycle=} {cycles=}")
                    print("SP CONTROL 1, generation 4")
                elif seq_method_i=='BADGEsubsel':
                    print("SP CONTROL 1, generation 3")
                    #proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                        mut_per_cycle, cycles=cycles, 
                    #                        ranker=ranker_sp) # pre Feb 22 2024
                    proposed_X=sp.generate_batch(n_to_make, X_source, 
                                            mut_per_cycle, cycles=cycles, 
                                            ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    print(f"First time a proposed_X ever appears (mutation): {len(proposed_X)=} {n_to_make=} {mut_per_cycle=} {cycles=}")
                    print("SP CONTROL 1, generation 4")
                elif seq_method_i=='LCMDsubsel':
                    print("SP CONTROL 1, generation 3")
                    #proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                        mut_per_cycle, cycles=cycles, 
                    #                        ranker=ranker_sp) # pre Feb 22 2024
                    proposed_X=sp.generate_batch(n_to_make, X_source, 
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    print(f"First time a proposed_X ever appears (realmut): {len(proposed_X)=} {n_to_make=} {mut_per_cycle=} {cycles=}")
                    print("SP CONTROL 1, generation 4")
                elif seq_method_i=='BADGEfromt':
                    proposed_X=sp.generate_batch(n_to_make, ranker=ranker_sp)
                    print(f"First time a proposed_X ever appears (tot rand): {len(proposed_X)=}")

                elif seq_method_i=='LCMDfromt':
                    proposed_X=sp.generate_batch(n_to_make, ranker=ranker_sp)
                    print(f"First time a proposed_X ever appears (tot rand): {len(proposed_X)=}")
                elif seq_method_i=='LCMDfromd':
                    # def dinuc(self, x_source, n_to_make, cycles=1, batch_size=100): #AC
                    proposed_X=sp.generate_batch(X_source, n_to_make) #, cycles=cycles) #, batch_size=self.batch_size)
                elif seq_method_i=='LCMDfromJ':
                    print(f"DEBUG LCMDfromJ {type(X_source)=}")
                    #X_source5=torch.cat((X_source,X_source,X_source,X_source,X_source))
                    X_source5=np.concatenate((X_source,X_source,X_source,X_source,X_source))
                    proposed_X=sp.generate_batch(n_to_make, X_source5, mut_per_cycle, cycles, ranker_sp, temp)

                elif seq_method_i=='concatBADGE1' or seq_method_i=='concatrand1' or seq_method_i=='concatLCMD1' or seq_method_i=='concatTEMPABADGE1' or seq_method_i=='concatTEMPArand1' or seq_method_i=='concatTEMPALCMD1': 
                    #n_to_make1=int(n_to_make/3)
                    n_to_make1=n_to_make
                    sp1 = SequenceProposer(generation_method='totally_random', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    proposed_X_1=sp1.random_sampler(n_to_make1, ranker=ranker_sp)
                    sp2 = SequenceProposer(generation_method='mutation', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    # proposed_X_2=sp2.mutate_randomly(n_to_make1, X_source, 
                    #                         mut_per_cycle, cycles=cycles, 
                    #                         ranker=ranker_sp)
                    proposed_X_2=sp2.generate_batch(n_to_make1, X_source, 
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp)
                    sp3 = SequenceProposer(generation_method='saliency', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    #proposed_X_3=sp3.mutate_by_saliency(n_to_make1, X_source, cycles, mut_per_cycle, ranker_sp, temp) 
                    proposed_X_3=sp3.generate_batch(n_to_make1, X_source, mut_per_cycle, cycles, ranker_sp, temp)
                    print(f"{seq_method_i} DEBUG: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X=torch.empty(0)
                    proposed_X=torch.cat((proposed_X,proposed_X_1),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_2),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_3),axis=0)

                elif seq_method_i=='Costmixrand1' or seq_method_i=='CostmixLCMD1' or seq_method_i=='CostmixTEMPArand1' or seq_method_i=='CostmixTEMPALCMD1': 
                    #n_to_make1=int(n_to_make/3)
                    n_to_make_sal=10000 ##int(n_to_make/2)
                    n_to_make_rnd=25000 ##int(n_to_make*5/2)
                    sp1 = SequenceProposer(generation_method='totally_random', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    proposed_X_1=sp1.random_sampler(n_to_make_rnd, ranker=ranker_sp)

                    #wh=np.random.choice(np.arange(3),size=len(X_source),replace=True)
                    #ind_rnd=np.random.choice(np.arange(len(X_source)), size=n_to_make_rnd, replace=False)
                    #ind_sal=np.random.choice( np.array(list(set(np.arange(len(X_source)))-set(ind_rnd))) , size=n_to_make_sal, replace=False)
                    sp2 = SequenceProposer(generation_method='mutation', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    # proposed_X_2=sp2.mutate_randomly(n_to_make1, X_source, 
                    #                         mut_per_cycle, cycles=cycles, 
                    #                         ranker=ranker_sp)
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_rnd,replace=False)], # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.where(wh==1)], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[ind_rnd], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    X_source_twice=np.concatenate((X_source,X_source)) #X_source is smaller than n_to_make_rnd!
                    proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source_twice[np.sort(np.random.permutation(np.arange(len(X_source_twice))))][:n_to_make_rnd],
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp)
                    
                    sp3 = SequenceProposer(generation_method='saliency', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    #proposed_X_3=sp3.mutate_by_saliency(n_to_make1, X_source, cycles, mut_per_cycle, ranker_sp, temp) 
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_sal,replace=False)], mut_per_cycle, cycles, ranker_sp, temp) # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.where(wh==2)], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #It's ok if there are overlaps, since we must go into larger than 20,000 anyway
                    proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.sort(np.random.permutation(np.arange(len(X_source))))], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source, mut_per_cycle, cycles, ranker_sp, temp)

                    print(f"{seq_method_i} DEBUG: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X=torch.empty(0)
                    proposed_X=torch.cat((proposed_X,proposed_X_1),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_2),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_3),axis=0)

                elif seq_method_i=='PriceHundredLCMD': 
                    sp1 = SequenceProposer(generation_method='totally_random', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    proposed_X_1=sp1.random_sampler(100000, ranker=ranker_sp)

                    #wh=np.random.choice(np.arange(3),size=len(X_source),replace=True)
                    #ind_rnd=np.random.choice(np.arange(len(X_source)), size=n_to_make_rnd, replace=False)
                    #ind_sal=np.random.choice( np.array(list(set(np.arange(len(X_source)))-set(ind_rnd))) , size=n_to_make_sal, replace=False)
                    sp2 = SequenceProposer(generation_method='mutation', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    # proposed_X_2=sp2.mutate_randomly(n_to_make1, X_source, 
                    #                         mut_per_cycle, cycles=cycles, 
                    #                         ranker=ranker_sp)
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_rnd,replace=False)], # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.where(wh==1)], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[ind_rnd], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #X_source_twice=np.concatenate((X_source,X_source)) #X_source is smaller than n_to_make_rnd!
                    proposed_X_2=sp2.generate_batch(100000, np.concatenate((X_source,X_source,X_source,X_source,X_source)), #X_source_twice[np.sort(np.random.permutation(np.arange(len(X_source_twice))))][:n_to_make_rnd],
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp)
                    
                    sp3 = SequenceProposer(generation_method='saliency', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    #proposed_X_3=sp3.mutate_by_saliency(n_to_make1, X_source, cycles, mut_per_cycle, ranker_sp, temp) 
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_sal,replace=False)], mut_per_cycle, cycles, ranker_sp, temp) # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.where(wh==2)], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #It's ok if there are overlaps, since we must go into larger than 20,000 anyway
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.sort(np.random.permutation(np.arange(len(X_source))))], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source, mut_per_cycle, cycles, ranker_sp, temp)
                    proposed_X_3=sp3.generate_batch(100000, np.concatenate((X_source,X_source,X_source,X_source,X_source)), mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3

                    new_batch_indexes_batched_sp_1=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_1), n_to_make=25000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    new_batch_indexes_batched_sp_2=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_2), n_to_make=25000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    new_batch_indexes_batched_sp_3=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_3), n_to_make=10000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    print(f"CBSajscankAJ {seq_method_i} {len(new_batch_indexes_batched_sp_1)=} {len(new_batch_indexes_batched_sp_2)=} {len(new_batch_indexes_batched_sp_3)=}")
                    print(f"CBSajscankAJ {seq_method_i} PRE: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X_1=proposed_X_1[new_batch_indexes_batched_sp_1]
                    proposed_X_2=proposed_X_2[new_batch_indexes_batched_sp_2]
                    proposed_X_3=proposed_X_3[new_batch_indexes_batched_sp_3]
                    print(f"CBSajscankAJ {seq_method_i} POST: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X=torch.empty(0)
                    proposed_X=torch.cat((proposed_X,proposed_X_1),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_2),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_3),axis=0)
                elif seq_method_i=='Price20KLCMD': 
                    sp1 = SequenceProposer(generation_method='totally_random', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    proposed_X_1=sp1.random_sampler(20000, ranker=ranker_sp)

                    #wh=np.random.choice(np.arange(3),size=len(X_source),replace=True)
                    #ind_rnd=np.random.choice(np.arange(len(X_source)), size=n_to_make_rnd, replace=False)
                    #ind_sal=np.random.choice( np.array(list(set(np.arange(len(X_source)))-set(ind_rnd))) , size=n_to_make_sal, replace=False)
                    sp2 = SequenceProposer(generation_method='mutation', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    # proposed_X_2=sp2.mutate_randomly(n_to_make1, X_source, 
                    #                         mut_per_cycle, cycles=cycles, 
                    #                         ranker=ranker_sp)
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_rnd,replace=False)], # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[np.where(wh==1)], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_2=sp2.generate_batch(n_to_make_rnd, X_source[ind_rnd], #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #X_source_twice=np.concatenate((X_source,X_source)) #X_source is smaller than n_to_make_rnd!
                    proposed_X_2=sp2.generate_batch(20000, X_source, #X_source_twice[np.sort(np.random.permutation(np.arange(len(X_source_twice))))][:n_to_make_rnd],
                                            cycles, cycles=mut_per_cycle, 
                                            ranker=ranker_sp)
                    
                    sp3 = SequenceProposer(generation_method='saliency', #SEQMETHOD0
                                    sequence_length=self.updated_full_X.shape[-1], seed=self.seed_add+222*self.current_i_AL, #seed=self.seed_add, 
                                    track_time=False, track_uncertanties = False, 
                                    track_batches=False, track_hamming=False, 
                                    track_pref = self.outdir+self.outflag) 
                    #proposed_X_3=sp3.mutate_by_saliency(n_to_make1, X_source, cycles, mut_per_cycle, ranker_sp, temp) 
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.random.choice(np.arange(len(X_source)),size=n_to_make_sal,replace=False)], mut_per_cycle, cycles, ranker_sp, temp) # AC this may risk having overlapped X_source sequences among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.where(wh==2)], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #It's ok if there are overlaps, since we must go into larger than 20,000 anyway
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source[np.sort(np.random.permutation(np.arange(len(X_source))))], mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3
                    #proposed_X_3=sp3.generate_batch(n_to_make_sal, X_source, mut_per_cycle, cycles, ranker_sp, temp)
                    proposed_X_3=sp3.generate_batch(20000, X_source, mut_per_cycle, cycles, ranker_sp, temp) #AC this way there is no overlap among proposed_X_1, proposed_X_2, and proposed_X_3

                    new_batch_indexes_batched_sp_1=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_1), n_to_make=5000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    new_batch_indexes_batched_sp_2=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_2), n_to_make=5000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    new_batch_indexes_batched_sp_3=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), 
                                                                                             x_pool=torch.tensor(proposed_X_3), n_to_make=2000, #self.how_many_new_batches_at_once*self.batch_size, 
                                                                                             models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                    print(f"CBSajscankAJ {seq_method_i} {len(new_batch_indexes_batched_sp_1)=} {len(new_batch_indexes_batched_sp_2)=} {len(new_batch_indexes_batched_sp_3)=}")
                    print(f"CBSajscankAJ {seq_method_i} PRE: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X_1=proposed_X_1[new_batch_indexes_batched_sp_1]
                    proposed_X_2=proposed_X_2[new_batch_indexes_batched_sp_2]
                    proposed_X_3=proposed_X_3[new_batch_indexes_batched_sp_3]
                    print(f"CBSajscankAJ {seq_method_i} POST: {proposed_X_1.shape=} {proposed_X_2.shape=} {proposed_X_3.shape=}")
                    proposed_X=torch.empty(0)
                    proposed_X=torch.cat((proposed_X,proposed_X_1),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_2),axis=0)
                    proposed_X=torch.cat((proposed_X,proposed_X_3),axis=0)


                elif seq_method_i=='saliency':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='realsal' or seq_method_i=='realTEMPAsal':
                    #    def mutate_randomly(   self, n_to_make, x_source, mutation_number, cycles=1, ranker=None):
                    #    def mutate_by_saliency(self, n_to_make, x_source, cycles, mutations_per, ranker, temp, decay=None): #, to_backprop='unc'):
                    # proposed_X=sp.generate_batch(n_to_make, X_source, 
                    #                         cycles, cycles=mut_per_cycle, 
                    #                         ranker=ranker_sp) # otherwise the random mutations will be exactly the same for every AL cycle (although different from the initial dataset)
                    if not (args.nickname in ['JRZJMzvvO','JRZJMzv5O','QRZJMzvvO','QRZJMzv5O','DRZJMzvvO','DRZJMzv5O']):
                        proposed_X=sp.generate_batch(n_to_make, X_source, mut_per_cycle, cycles, ranker_sp, temp)
                    else:
                        X_source5=np.concatenate((X_source,X_source,X_source,X_source,X_source))
                        proposed_X=sp.generate_batch(n_to_make, X_source5, mut_per_cycle, cycles, ranker_sp, temp)
                elif seq_method_i=='saliency_y':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='saliency_div_y':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='saliency_U-A':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='saliency_aleat':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='saliency_evidential':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='GradientSHAP':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                elif seq_method_i=='DeepLiftSHAP':
                    proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)

                elif seq_method_i=='simulated_annealing':
                    proposed_X=sp.generate_batch(n_to_make, X_source, self.batch_size, mut_per_cycle, 
                                        cycles, temp, decay, ranker_sp, 
                                        min_frac=0, min_abs= 0, prevent_decreases=False)
                elif seq_method_i=='simulated_annealing_y':
                    proposed_X=sp.generate_batch(n_to_make, X_source, self.batch_size, mut_per_cycle, 
                                        cycles, temp, decay, ranker_sp, 
                                        min_frac=0, min_abs= 0, prevent_decreases=False)
                    
                elif seq_method_i=='greedy':
                    proposed_X=sp.generate_batch(n_to_make, X_source, self.batch_size, cycles, ranker_sp)
                elif seq_method_i=='greedy_y':
                    proposed_X=sp.generate_batch(n_to_make, X_source, self.batch_size, cycles, ranker_sp)
                    
                elif seq_method_i=='genetic':
                    proposed_X=sp.generate_batch(n_to_make, X_source, mut_per_cycle, cycles, 
                                    x_prob, expansion_fold, ranker_sp)
                elif seq_method_i=='genetic_y':
                    proposed_X=sp.generate_batch(n_to_make, X_source, mut_per_cycle, cycles, 
                                    x_prob, expansion_fold, ranker_sp)

                elif seq_method_i=='fromfile':
                    if self.chosen_model=='ResidualBind': #QUIQUIURG this should not be an if, but should apply to any model
                        if 'VTS1' in args.chosen_dataset: #QUIQUINONURG remove args
                            file_to_open='./inputs/VTS1_rnacompete2009_processed_for_dal.h5'
                        elif 'RBFOX1' in args.chosen_dataset: #QUIQUINONURG remove args
                            file_to_open='./inputs/RBFOX1_rnacompete2013_processed_for_dal.h5'
                        elif 'LentiMPRA' in args.chosen_dataset: #QUIQUINONURG remove args
                            file_to_open='./inputs/LentiMPRA_processed_for_dal.h5'
                        else:
                            print("ERROR: wrongly selected dataset for fromfile method")
                            exit()
                    elif self.chosen_model=='DeepSTARR':
                        file_to_open='./inputs/DeepSTARRdev.h5'
                    elif self.chosen_model=='NewResNet' or self.chosen_model=='LegNetPK':
                        if 'newLentiMPRAK562' in args.chosen_dataset: #QUIQUINONURG remove args
                            file_to_open='./inputs/newLentiMPRAK562_processed_for_dal.h5'
                        elif 'newLentiMPRAHepG2' in args.chosen_dataset: #QUIQUINONURG remove args
                            file_to_open='./inputs/newLentiMPRAHepG2_processed_for_dal.h5'
                        else:
                            print("ERROR: wrongly selected dataset for fromfile method")
                            exit()
                    else:
                        print("ERROR: wrongly selected model for fromfile method")
                        exit()
                    if self.uncertainty_method=='no': #QUIQUINONURG can this be a little more precise?
                        #load_from_file=True
                        always_from_zero=False
                        path_to_alr_inds=self.outdir+'fromfile_alrinds_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'
                    else:
                        #load_from_file=False #AC needs to be this way otherwise the file will store all sequences, not just the subselected ones.
                        always_from_zero=True #AC needs to be this way otherwise the file will store all sequences, not just the subselected ones.
                        path_to_alr_inds=self.outdir+'fromfile_useless_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'
                    #print(f"FROMFILE: {always_from_zero=}")
                    path_to_alr_inds_AL0='./inputs/usables_'+args.chosen_dataset+'.npy'
                    print(f"A2CHEKCFROMFILE: in DAL Pipeline: {self.current_i_AL=} {path_to_alr_inds_AL0=} {path_to_alr_inds=} {always_from_zero=}")
                    #proposed_X=sp.generate_batch(self.chosen_model, file_to_open, n_to_make, self.updated_full_X) # it MUST be updated_full_X: at later iALs, you may have already accessed the pool.
                    #proposed_X=sp.generate_batch(self.chosen_model, file_to_open, n_to_make, self.updated_full_X, method='random', load_from_file=True, ALcycle=self.current_i_AL, path_to_alr_inds=self.outdir+'fromfile_alrinds_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy') # it MUST be updated_full_X: at later iALs, you may have already accessed the pool.
                    proposed_X=sp.generate_batch(self.chosen_model, file_to_open, n_to_make, ALcycle=self.current_i_AL, always_from_zero=always_from_zero,
                         path_to_alr_inds=path_to_alr_inds, path_to_alr_inds_AL0=path_to_alr_inds_AL0)

                elif seq_method_i=='dinuc':
                    # def dinuc(self, x_source, n_to_make, cycles=1, batch_size=100): #AC
                    proposed_X=sp.generate_batch(X_source, n_to_make) #, cycles=cycles) #, batch_size=self.batch_size)

                elif seq_method_i=='evoaug':
                #elif 'evoaug' in seq_method_i:
                    proposed_X=sp.generate_batch(X_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=mutant_fraction)
                elif seq_method_i=='evoaugmut':
                    proposed_X=sp.generate_batch(X_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=mutant_fraction)
                elif seq_method_i=='evoaugassign':
                    proposed_X=sp.generate_batch(X_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=mutant_fraction)
                elif seq_method_i=='realevoaug':
                    proposed_X=sp.generate_batch(X_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=mutant_fraction)

                elif seq_method_i=='motifembed':
                    """
                    FIMO scans for LentiMPRA motifs: KLF5, KLF15, NFYA, NFYC, FOXI1, FOXJ2.
                    HepG2: HNF4A, HNF4G. K562: GATA2, GATA3.
                    """
                    pfm_file_path='./pfm_AC_all.txt'
                    if 'K562' in args.chosen_dataset:
                        seq_length=230
                        #core_names=['KLF5', 'KLF15', 'NFYA', 'NFYC', 'FOXI1', 'FOXJ2']
                        core_names=['KLF5', 'KLF15', 'NFYA', 'NFYC', 'FOXI1', 'FOXJ2','GATA2', 'GATA3']
                    elif 'HepG2' in args.chosen_dataset:
                        seq_length=230
                        core_names=['KLF5', 'KLF15', 'NFYA', 'NFYC', 'FOXI1', 'FOXJ2','HNF4A', 'HNF4G']
                    else:
                        print("ERROR: wrongly selected dataset for motif embedding.")
                    ##proposed_X=sp.generate_batch(n_to_make,seq_length,pfm_file_path,core_names)
                    proposed_X=sp.generate_batch(X_source,pfm_file_path,core_names)

                elif seq_method_i=='vanilla_diffusion':
                    """
                    if 'VTS1' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/VTS1_rnacompete2009_processed_for_dal.h5'
                    elif 'RBFOX1' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/RBFOX1_rnacompete2013_processed_for_dal.h5'
                    elif 'LentiMPRA_' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/LentiMPRA_processed_for_dal.h5'
                        #diffmodel_path='../D3-DNA-Discrete-Diffusion/Training\ and\ Sampling/exp_local/deepstarr/2024.06.10/111245/checkpoints-meta/checkpoint.pth'
                        diffmodel_path='../inputs/diffmodel_checkpoint-meta.pth'
                    elif 'LentiMPRA_' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/LentiMPRA_processed_for_dal.h5'
                        #diffmodel_path='../D3-DNA-Discrete-Diffusion/Training\ and\ Sampling/exp_local/deepstarr/2024.06.10/111245/checkpoints-meta/checkpoint.pth'
                        diffmodel_path='../inputs/diffmodel_checkpoint-meta.pth'
                    elif 'newLentiMPRAK562' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/newLentiMPRAK562_processed_for_dal.h5'
                        #diffmodel_path='../inputs/diffmodel_checkpoint-meta_newLentiMPRAK562_processed_for_dal.pth'
                        #diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_25K_random/2024.06.19/165411' #/checkpoints-meta/checkpoint.pth'
                        diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_25K_random/2024.06.24/114726' 
                    elif 'newLentiMPRAHepG2' in args.chosen_dataset: #QUIQUINONURG remove args
                        file_to_open='./inputs/newLentiMPRAHepG2_processed_for_dal.h5'
                        #diffmodel_path='../inputs/diffmodel_checkpoint-meta_newLentiMPRAHepG2_processed_for_dal.pth'
                    else:
                        print("ERROR: wrong dataset selected for vanilla diffusion")
                        exit()
                    """
                    if 'newLentiMPRAK562' in args.chosen_dataset and '20000' in args.chosen_dataset and 'random0' in args.chosen_dataset and args.chosen_model=='LegNetPK': #QUIQUINONURG remove args and make it more generalizable
                        ##diffmodel_path='../inputs/diffmodel_checkpoint-meta_newLentiMPRAK562_processed_for_dal.pth'
                        ##diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_25K_random/2024.06.19/165411' #/checkpoints-meta/checkpoint.pth'
                        ##diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_25K_random/2024.06.24/114726'
                        diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/LegNetPK_K562_20K_random/2024.07.03/182846'
                        seqslength=230
                    elif 'newLentiMPRAK562' in args.chosen_dataset and '130000' in args.chosen_dataset and 'random0' in args.chosen_dataset and args.chosen_model=='LegNetPK': #QUIQUINONURG remove args
                        diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/LegNetPK_K562_130K_random/2024.07.06/092504' #DONE
                        seqslength=230
                    #elif 'newLentiMPRAK562' in args.chosen_dataset and args.chosen_model=='NewResNet': #QUIQUINONURG remove args
                    #    #diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_25K_random/2024.06.24/114726'
                    #    diffmodel_path='../D3-DNA-Discrete-Diffusion/Training and Sampling/exp_local/NewResNet_K562_20K_random/2024.06.30/090948'
                    #    seqslength=230
                    else:
                        print("ERROR: wrong dataset selected for vanilla diffusion")
                        exit()
                    ##proposed_X=sp.generate_batch(model_path=diffmodel_path, h5file=file_to_open, steps=128, batch_size=n_to_make)
                    proposed_X=sp.generate_batch(model_path=diffmodel_path, n_to_make=n_to_make, ranker=ranker_sp, seqlength=seqslength, steps=128)
                
                elif seq_method_i=='diffusion_file': #V
                    if 'newLentiMPRAK562' in args.chosen_dataset and '20000' in args.chosen_dataset and 'random0' in args.chosen_dataset and args.chosen_model=='LegNetPK': #QUIQUINONURG remove args
                        file_to_open='./inputs/'+args.chosen_dataset+'.h5' ##'./inputs/newLentiMPRAK562_processed_for_dal.h5'
                        diffmodel_path='/home/crnjar/D3-DNA-Discrete-Diffusion/Training_and_Sampling_Conditioned/exp_local/LegNet_K562_20K_random_cond/2024.07.04/100518' #DONE
                        seqslength=230
                    elif 'newLentiMPRAK562' in args.chosen_dataset and '130000' in args.chosen_dataset and 'random0' in args.chosen_dataset and args.chosen_model=='LegNetPK': #QUIQUINONURG remove args
                        file_to_open='./inputs/'+args.chosen_dataset+'.h5' ##'./inputs/newLentiMPRAK562_processed_for_dal.h5'
                        diffmodel_path='/home/crnjar/D3-DNA-Discrete-Diffusion/Training_and_Sampling_Conditioned/exp_local/LegNet_K562_130K_random_cond/2024.07.07/212702' #DONE
                        seqslength=230

                    proposed_X=sp.generate_batch(model_path=diffmodel_path, n_to_make=n_to_make, h5file=file_to_open, ranker=ranker_sp, seqlength=seqslength, steps=128)

                elif seq_method_i=='diffusion_y': #V unless we want to use a dynamic quantile
                    if 'newLentiMPRAK562' in args.chosen_dataset and args.chosen_model=='LegNetPK': #QUIQUINONURG remove args
                        ##file_to_open=chosen_dataset.replace('random0','mostac').replace('leastac','mostac')
                        diffmodel_path='/home/crnjar/D3-DNA-Discrete-Diffusion/Training_and_Sampling_Conditioned/exp_local/LegNet_K562_20K_random_cond/2024.07.04/100518'
                        ycond=0.41 #0.85 quantile of random0 20000 # quantile_of_random0.py
                        seqslength=230
                    proposed_X=sp.generate_batch(model_path=diffmodel_path, n_to_make=n_to_make, ranker=ranker_sp, seqlength=seqslength, steps=128, ycond=ycond)

                elif seq_method_i=='hessian':
                    print("SP CONTROL 1, generation 3 hess")
                    ##proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                    ##proposed_X=sp.generate_batch(n_to_make, X_source, cycles, mut_per_cycle, ranker_sp, temp)
                    #proposed_X=sp.generate_batch(n_to_make, X_source, X_prev, cycles, mut_per_cycle, ranker_sp, temp)
                    proposed_X=sp.generate_batch(n_to_make, X_source, None, cycles, mut_per_cycle, ranker_sp, temp) #QUIQUIURG should impose to be different than the previous?
                    print("SP CONTROL 1, generation 4 hess")
                else:
                    #print('Selected sequence proposal method: ',seq_method_i)
                    print("Error: Method for sequence proposing wrongly selected in propose_new_sequences:",seq_method_i)
                    exit()

                if self.screm:
                    proposed_X=torch.tensor(np.array(set_torch_tensors_test.lookup_dna_to_ohe(list(set(set_torch_tensors_test.lookup_ohe_to_dna(proposed_X))-set(set_torch_tensors_test.lookup_ohe_to_dna(self.updated_full_X))))),dtype=torch.float32) 
                    print(f"SCREM: new shape after reducing: {proposed_X.shape=}") #added 23 october 2024

                print("SP CONTROL 1, generation 5")
                ##proposed_y=self.Oracle_Model.predict_custom(proposed_X)
                proposed_y=self.Oracle_Model.interrogate(proposed_X)
                proposed_y = proposed_y.detach().cpu().numpy()
                if seq_method_i=='evoaugassign':
                    if len(X_source)!=self.generated_U:
                        print("ERROR: y_source would not correspond to X_source since a shuffle would take place") #QUIQUIURG or maybe this would not be a problem if we never get into pool_gen!
                    y_source = self.orig_data_module['Y_train']
                    proposed_y=y_source
                proposed_X = proposed_X.detach().cpu().numpy()
                no_sequences = False
                randinds = np.array([])
                print(f"SP CONTROL {no_sequences=} ")
                print(f"SP CONTROL {randinds=}")
                print(f"SP CONTROL {proposed_X.shape=}")
                print(f"SP CONTROL {proposed_y.shape=}")

                """
                # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB (GPU 0; 31.74 GiB total capacity; 31.19 GiB already allocated; 11.38 MiB free; 31.24 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
                unc_all, preds_av_ = ranker_sp.calculate_desiderata(proposed_X, keep_grads = True)
                np.save(self.outdir+'uncall_'+self.nickname+'_seedadd-'+str(self.seed_add)+'_iAL-'+str(self.current_i_AL)+'.npy',np.array(unc_all))
                """


            
        else:
            #print("SP CONTROL 1, else")
            print("Error Method for sequence proposing wrongly selected from list:",self.seq_method)
            exit()
        
        if no_sequences:
            randinds, proposed_X, proposed_y = None, None, None
        return randinds, proposed_X, proposed_y, no_sequences 

    def data_pruning(self):
        """ In here you must act upon self.updated_full_X, given at what point it is placed."""
        pass

    def Active_Learning_Cycle_Part_2(self):
        """ Second part : make new sequences for ranking with newly-trained model """
        #no_sequences=False

        #print("CONTROL: 1")

        # Re-loading 
        if args.what_part!='all': self.load_for_Part_2() #QUIQUINONURG this lines makes cumulative metrics only accessible if you do not use what_part==all

        set_random_seed(self.seed_add) # QUIQUIURG this is needed in case we do not evoke the Active_Learning_Loop function, as it is set at its beginning. But is it ok to set it here too?
        if self.seq_method!='backprop-gen-and-rank':
            if len(self.already)!=len(np.unique(self.already)): 
                print("list already contains some indexes multiple times!")
                exit()

            if args.selection_first=='no':
                #print("CONTROL: 1a")
                randinds, proposed_X, proposed_y, no_sequences = self.propose_new_sequences() #no_sequences)
                #print("CONTROL: 1b")

                # Perform ranking of proposed_X
                if not no_sequences:
                    if self.diversity_method=='power':
                        ranker_rk = PowerRanker(self.DAL_Models,
                                            self.batch_size,
                                            how_many_batches=self.how_many_new_batches_at_once,
                                            beta = self.uncertainty_weight,
                                            uncertainty_method='no',
                                            diversity_method=self.diversity_method,
                                            highpred_method=self.highpred_method,
                                            uncertainty_weight=0.0,
                                            diversity_weight=self.diversity_weight,
                                            highpred_weight=self.highpred_weight,
                                            chosen_model=self.chosen_model,
                                            cycle=self.current_i_AL+1,
                                            sigmadistr_freq=self.sigmadistr_freq,
                                            outdir=self.outdir,
                                            outflag=self.outflag+'_seedadd-'+str(self.seed_add),  #QUIQUINONURG should this outflag be changed to nickname?
                                            device=device,
                                            local_seed=self.seed_add, 
                                            task_type=self.task_type)
                        
                    elif self.diversity_method=='softmax':
                        ranker_rk = SoftmaxRanker(self.DAL_Models,
                                            self.batch_size,
                                            how_many_batches=self.how_many_new_batches_at_once,
                                            beta = self.uncertainty_weight,
                                            uncertainty_method='no',
                                            diversity_method=self.diversity_method,
                                            highpred_method=self.highpred_method,
                                            uncertainty_weight=0.0,
                                            diversity_weight=self.diversity_weight,
                                            highpred_weight=self.highpred_weight,
                                            chosen_model=self.chosen_model,
                                            cycle=self.current_i_AL+1,
                                            sigmadistr_freq=self.sigmadistr_freq,
                                            outdir=self.outdir,
                                            outflag=self.outflag+'_seedadd-'+str(self.seed_add),  #QUIQUINONURG should this outflag be changed to nickname?
                                            device=device,
                                            local_seed=self.seed_add, 
                                            task_type=self.task_type)
                        
                    elif self.diversity_method=='softrank':
                        ranker_rk = SoftrankRanker(self.DAL_Models,
                                            self.batch_size,
                                            how_many_batches=self.how_many_new_batches_at_once,
                                            beta = self.uncertainty_weight,
                                            uncertainty_method='no',
                                            diversity_method=self.diversity_method,
                                            highpred_method=self.highpred_method,
                                            uncertainty_weight=0.0,
                                            diversity_weight=self.diversity_weight,
                                            highpred_weight=self.highpred_weight,
                                            chosen_model=self.chosen_model,
                                            cycle=self.current_i_AL+1,
                                            sigmadistr_freq=self.sigmadistr_freq,
                                            outdir=self.outdir,
                                            outflag=self.outflag+'_seedadd-'+str(self.seed_add),  #QUIQUINONURG should this outflag be changed to nickname?
                                            device=device,
                                            local_seed=self.seed_add, 
                                            task_type=self.task_type)
                    else: # STANDARD CASE (POOL BASED OR NOT)
                        """
                        Cases:
                        -A: Pool based random(no,no,no), generated_U == 
                        -B: Pool based unc rank, generated_U ==
                        -C: Pool based random(no,no,no), generated_U != 
                        -D: Pool based unc rank, generated_U !=

                        -E: MQS random(no,no,no), generated_U == 
                        -F: MQS based unc rank, generated_U ==
                        -G: MQS random(no,no,no), generated_U != 
                        -H: MQS unc rank, generated_U !=
                        """

                        #print("CONTROL: 1c")
                        #if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no'): # ok 4 B,D,F,H
                        #    if not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size):
                        if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no') and not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size): #QUIQUIURG this should be the definitive, but it may be problematic for pool based
                            print(f"A2CHEKC - ranker_rk will be defined {self.uncertainty_method=} {self.diversity_method=} {self.highpred_method=} {self.generated_U=} {self.how_many_new_batches_at_once=} {self.batch_size=}")
                            ranker_rk = Ranker(self.DAL_Models,
                                        self.batch_size,
                                        how_many_batches=self.how_many_new_batches_at_once,
                                        uncertainty_method=self.uncertainty_method,
                                        diversity_method=self.diversity_method,
                                        highpred_method=self.highpred_method,
                                        uncertainty_weight=self.uncertainty_weight,
                                        diversity_weight=self.diversity_weight,
                                        highpred_weight=self.highpred_weight,
                                        chosen_model=self.chosen_model,
                                        cycle=self.current_i_AL+1,
                                        sigmadistr_freq=self.sigmadistr_freq,
                                        outdir=self.outdir,
                                        outflag=self.outflag+'_seedadd-'+str(self.seed_add),  #QUIQUINONURG should this outflag be changed to nickname?
                                        device=device,
                                        local_seed=self.seed_add, 
                                        task_type=self.task_type)
                    
                    if (self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor'): # POOL BASED # the only new line
                        #if not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size):
                        if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no'): # ok 4 B,D,F,H
                        #if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no') and not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size): 
                            print(f"A2CHEKC - CONTROL: 1d {self.uncertainty_method=} {self.diversity_method=} {self.highpred_method=} {self.generated_U=} {self.how_many_new_batches_at_once=} {self.batch_size=}")
                            new_batch_indexes_batched,cum_perc_unc,unc_thr=ranker_rk.rank(proposed_X, proposed_y.squeeze(axis=1)) #QUIQUIURG this currently ranks everything that comes in with no_sequences set to False TODO: figure out how this should be handled
                            #print("CONTROL: 1e")
                            self.monitored_part2['cum_perc_uncs'].append(cum_perc_unc)
                            self.monitored_part2['unc_thrs'].append(unc_thr)
                        else: # RANDOM RANKING (POOL BASED OR NOT)
                            print(f"A2CHEKC - RANDOM RANKING (POOL BASED OR NOT) dskjnjds {self.uncertainty_method=} {self.diversity_method=} {self.highpred_method=} {self.generated_U=} {self.how_many_new_batches_at_once=} {self.batch_size=}")
                            #if not (self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor'):
                            #    print("Got into poolbased-only section, but seq_method is not a poolbased one.")
                            #    exit()
                            ##new_batch_indexes_batched=list(np.random.choice(range(len(proposed_X)),size=self.how_many_new_batches_at_once*self.batch_size)) #QUIQUIURG size allows for repetitions!!! #QUIQUINONURG len(proposed_X) e giusto vero? - dovrebbe esserlo, si
                            #new_batch_indexes_batched=randinds
                            #print(f"{no_sequences=}")
                            new_batch_indexes_batched=np.arange(len(randinds)) #QUIQUIURGURG np.arange(len(proposed_X)) ???
                            np.random.shuffle(new_batch_indexes_batched) #QUIQUISOLVED? this is faster than random.shuffle() right?
                            new_batch_indexes_batched=new_batch_indexes_batched[:self.how_many_new_batches_at_once*self.batch_size] 
                        #else:
                        #    new_batch_indexes_batched=np.arange(len(proposed_X))

                    """
                    Got here, all letter cases were considered?
                    - new_batch_indexes_batched:   originally ABCDEFGH
                    - ranker_rk:                   originally BDFH
                    - .rank:                       originally BDFH
                    - randinds:                    originally before any
                    - proposed_X,y:                originally before any

                    
                    """

            elif args.selection_first=='yes':
                print('Error: selection_first==yes in development.')
                exit()
            else:
                print('Error: selection_first misassigned.')
                exit()

        else: # Generation and ranking performed together #WORKINPROGRESS 
            print("Generation and ranking from backpropagation under development.")
            print("Or else: can this be simplified, in the sense that it just spares to call the Highest ranking function?")
            exit()

        #print("CONTROL: 2")
        if no_sequences: print("WARNING: no sequences left to select. Part II of the DAL Cycle will be dummy.")

        if not no_sequences:
            if not (self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor'): # NOT POOL BASED
                if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no'):
                #if not (self.uncertainty_method=='no' and self.diversity_method=='no' and self.highpred_method=='no') and not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size): #pre 18Jan2024
                    if (self.generated_U==self.how_many_new_batches_at_once*self.batch_size): 
                        print("A2CHEKC This would have been affected by modification. Probably it was for speed reasons?")
                    print(f"A2CHEKC - ranker_rk will rank {self.uncertainty_method=} {self.diversity_method=} {self.highpred_method=} {self.generated_U=} {self.how_many_new_batches_at_once=} {self.batch_size=}")
                    new_batch_indexes_batched_sp,cum_perc_unc_sp,unc_thr_sp=ranker_rk.rank(proposed_X, proposed_y.squeeze(axis=1)) #QUIQUIURG this currently ranks everything that comes in with no_sequences set to False TODO: figure out how this should be handled
                    print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                    self.monitored_part2['cum_perc_uncs'].append(cum_perc_unc_sp)
                    self.monitored_part2['unc_thrs'].append(unc_thr_sp)

                    #AC there are cases with F that dont require subselection, so this part is optional. Being optional, it SHOULD be here, and not within the else of ref vfehuinCHUDA
                    #if self.seq_method=='fromfile': #last for BoG, it is from for methods K (fromfile+saliency)
                    if 'fromfile' in self.seq_method:
                        print("A2CHEKCFROMFILE will save new indexes for fromfile method")
                        path_to_alr_inds_AL0_inrank='./inputs/usables_'+args.chosen_dataset+'.npy'
                        #path_to_alr_inds_inrank=self.outdir+'fromfile_alrinds_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'
                        path_to_alr_inds_inrank=self.outdir+'fromfile_useless_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy'
                        loaded_indexes=np.load(path_to_alr_inds_AL0_inrank) 
                        ##i_to_keep=np.isin(loaded_indexes,loaded_indexes[new_batch_indexes_batched_sp],assume_unique=True,invert=True) #test_indexes_rank_fromfile.py
                        ##tosave_indexes=loaded_indexes[i_to_keep] #test_indexes_rank_fromfile.py
                        tosave_indexes=loaded_indexes[new_batch_indexes_batched_sp] #QUIQUIURGM should it be the whole thing instead?
                        print(f"A2CHEKCFROMFILE before concatenation: {len(tosave_indexes)=}")
                        if os.path.isfile(path_to_alr_inds_inrank):
                            print("A2CHEKCFROMFILE will concatenate with:",path_to_alr_inds_inrank)
                            past_removed=np.load(path_to_alr_inds_inrank)
                            tosave_indexes=np.concatenate((past_removed,tosave_indexes))
                            print(f"A2CHEKCFROMFILE {len(past_removed)=}")
                        print(f"A2CHEKCFROMFILE after concatenation with past_removed (fromfile_useless.npy): {len(tosave_indexes)=}")
                        np.save(path_to_alr_inds_inrank,tosave_indexes)

                    # if self.seq_method=='BatchBALDsubsel':
                    #     new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='maxdet') #QUIQUIURG not sure about x_train=self.updated_full_X
                    #     print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                    # if self.seq_method=='BADGEsubsel':
                    #     new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='kmeanspp') #QUIQUIURG not sure about x_train=self.updated_full_X
                    #     print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                else: #vfehuinCHUDA
                    print("A2CHEKC random ranking, not pool based")

                    if not (self.seq_method=='BatchBALDsubsel' or \
                            self.seq_method=='BADGEsubsel' or \
                            self.seq_method=='concatBADGE1' or \
                            self.seq_method=='concatrand1' or \
                            self.seq_method=='concatLCMD1' or \
                            self.seq_method=='BADGEfromt' or \
                            self.seq_method=='LCMDsubsel' or \
                            self.seq_method=='LCMDfromt' or \
                            self.seq_method=='LCMDfromJ' or \
                            self.seq_method=='LCMDfromd' or \
                            self.seq_method=='concatTEMPArand1' or \
                            self.seq_method=='concatTEMPALCMD1'):
                        print("NOT ANY OTHER METHOD:",self.seq_method)
                        #A2CHEKC qui dovrebbe essere come alle linee:
                        new_batch_indexes_batched_sp=np.arange(len(proposed_X)) #QUIQUIURGURG should be good, right?? or np.arange(genU)???? Or is it conceptually identical too?
                        np.random.shuffle(new_batch_indexes_batched_sp) #QUIQUISOLVED? this is faster than random.shuffle() right?
                        new_batch_indexes_batched_sp=new_batch_indexes_batched_sp[:self.how_many_new_batches_at_once*self.batch_size] 

                    elif self.seq_method=='BatchBALDsubsel':
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0].to('cuda')], y_train=self.updated_full_y, selection_method='maxdet',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='maxdet',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                    elif self.seq_method=='BADGEsubsel' or self.seq_method=='BADGEfromt':
                        print(f"DEBUGBADGEX: {proposed_X.shape=} {self.updated_full_X.shape=} {[self.DAL_Models[0]]=} {self.batch_size=} {self.updated_full_y=} {self.how_many_new_batches_at_once=}")
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0].to('cuda')], y_train=self.updated_full_y, selection_method='kmeanspp',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='kmeanspp',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                    elif self.seq_method=='LCMDsubsel' or self.seq_method=='LCMDfromt' or self.seq_method=='LCMDfromd' or self.seq_method=='LCMDfromJ':
                        print(f"DEBUGLCMDX: {proposed_X.shape=} {self.updated_full_X.shape=} {[self.DAL_Models[0]]=} {self.batch_size=} {self.updated_full_y=} {self.how_many_new_batches_at_once=}")
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0].to('cuda')], y_train=self.updated_full_y, selection_method='kmeanspp',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), x_pool=torch.tensor(proposed_X), n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")

                    elif self.seq_method=='concatBADGE1':
                        new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='kmeanspp',external_batch_size=self.batch_size) #QUIQUIURG not sure about x_train=self.updated_full_X
                        print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                        how_many_method_1=np.where((new_batch_indexes_batched_sp < n_to_make1))[0]
                        how_many_method_2=np.where((new_batch_indexes_batched_sp > n_to_make1) & (new_batch_indexes_batched_sp < 2*n_to_make1))[0]
                        how_many_method_3=np.where((new_batch_indexes_batched_sp > 2*n_to_make1))[0]
                        print(f"Stats of concatBADGE1: {len(how_many_method_1)=} {len(how_many_method_2)=} {len(how_many_method_3)=}")
                        final_proposed_X=proposed_X[new_batch_indexes_batched_sp] #AC This is not an error, should be here 
                        final_proposed_y=proposed_y[new_batch_indexes_batched_sp]
                    elif self.seq_method=='concatLCMD1' or self.seq_method=='concatTEMPALCMD1':
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=self.updated_full_X, x_pool=proposed_X, n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X).to('cuda'), x_pool=torch.tensor(proposed_X).to('cuda'), n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        #new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), x_pool=torch.tensor(proposed_X), n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0]], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        new_batch_indexes_batched_sp=bmdal_dholzmueller.batch_selection_method(x_train=torch.tensor(self.updated_full_X), x_pool=torch.tensor(proposed_X), n_to_make=self.how_many_new_batches_at_once*self.batch_size, models=[self.DAL_Models[0].to('cpu')], y_train=self.updated_full_y, selection_method='lcmd',external_batch_size=self.batch_size, base_kernel='grad', kernel_transforms=[('rp', [512])], sel_with_train=True) #QUIQUIURG not sure about x_train=self.updated_full_X
                        print(f"A2CHEKC {len(new_batch_indexes_batched_sp)=}")
                        n_to_make1=self.generated_U #QUIQUIURG why is this necessary? it shoudln't! n_to_make1 should be defined at this point!
                        how_many_method_1=np.where((new_batch_indexes_batched_sp < n_to_make1))[0]
                        how_many_method_2=np.where((new_batch_indexes_batched_sp > n_to_make1) & (new_batch_indexes_batched_sp < 2*n_to_make1))[0]
                        how_many_method_3=np.where((new_batch_indexes_batched_sp > 2*n_to_make1))[0]
                        print(f"Stats of concatLCMD1: {len(how_many_method_1)=} {len(how_many_method_2)=} {len(how_many_method_3)=}")
                        final_proposed_X=proposed_X[new_batch_indexes_batched_sp] #AC This is not an error, should be here 
                        final_proposed_y=proposed_y[new_batch_indexes_batched_sp]
                    elif self.seq_method=='concatrand1' or self.seq_method=='concatTEMPArand1':
                        #new_batch_indexes_batched=np.arange(len(randinds)) #QUIQUIURGURG np.arange(len(proposed_X)) ???
                        new_batch_indexes_batched_sp=np.arange(len(proposed_X)) #QUIQUIURGURG np.arange(len(proposed_X)) ???
                        np.random.shuffle(new_batch_indexes_batched_sp) #QUIQUISOLVED? this is faster than random.shuffle() right?
                        print(f"PREREDUCING: {len(new_batch_indexes_batched_sp)=}")
                        new_batch_indexes_batched_sp=new_batch_indexes_batched_sp[:self.how_many_new_batches_at_once*self.batch_size] 
                        print(f"POSTREDUCING: {len(new_batch_indexes_batched_sp)=}")
                        #n_to_make1=n_to_make #QUIQUIURG why was n_to_make not defined at this stage??? Doesnt make sense
                        n_to_make1=self.generated_U #QUIQUIURG why is this necessary? it shoudln't! n_to_make1 should be defined at this point!
                        how_many_method_1=np.where((new_batch_indexes_batched_sp < n_to_make1))[0]
                        how_many_method_2=np.where((new_batch_indexes_batched_sp > n_to_make1) & (new_batch_indexes_batched_sp < 2*n_to_make1))[0]
                        how_many_method_3=np.where((new_batch_indexes_batched_sp > 2*n_to_make1))[0]
                        print(f"Stats of concatrand1: {len(how_many_method_1)=} {len(how_many_method_2)=} {len(how_many_method_3)=}")
                        final_proposed_X=proposed_X[new_batch_indexes_batched_sp] #AC This is not an error, should be here 
                        final_proposed_y=proposed_y[new_batch_indexes_batched_sp]
                        print(f"POSTREDUCING1: {len(new_batch_indexes_batched_sp)=}")
                    else:
                        print("ERROR: no batch mode nor random subselection.")
                    
                if not (self.generated_U==self.how_many_new_batches_at_once*self.batch_size): #pre 18Jan2024 #AC 4 nov 2024 this does not include JRZ8 JRZA or JRZ7 JRZD
                    print("A2CHEKC not self genU ==  hmnbao *bs")
                    final_proposed_X=proposed_X[new_batch_indexes_batched_sp]
                    final_proposed_y=proposed_y[new_batch_indexes_batched_sp]
                else:
                    print("HHEHRJNI")
                    if not (self.seq_method=='concatrand1' or self.seq_method=='concatTEMPArand1' or self.seq_method=='concatLCMD1' or self.seq_method=='concatTEMPALCMD1'):
                        print("HHEHRJNI1")
                        print("A2CHEKC self genU ==  hmnbao *bs")
                        final_proposed_X=proposed_X
                        final_proposed_y=proposed_y
                    print(f"A2CHEKC - final_proposed_X=proposed_X {self.uncertainty_method=} {self.diversity_method=} {self.highpred_method=} {self.generated_U=} {self.how_many_new_batches_at_once=} {self.batch_size=}")
                # A2CHEKC IF  (self.generated_U==self.how_many_new_batches_at_once*self.batch_size):
                #                     final_proposed_X=proposed_X
            else:
                print("A2CHEKC this if has gone wrong.")

        print("CCC1",no_sequences)
        if not no_sequences:
            if self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor': # POOL BASED
                #print("CONTROL: 4")
                # Update indexes
                for ind in new_batch_indexes_batched:
                    #self.already.append(ind)
                    if self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor':
                        self.already.append(randinds[ind]) 
                    #else:
                    #    print("ERROR: have you checked that randins is defined for this seq method?????") #WORKINPROGRESS #DEBUG
                    #    exit()
                if len(self.already)!=len(np.unique(self.already)): 
                    print("ERROR! already array not unique!")
                    exit()
                # Updata data module
                #print(f"{self.updated_full_X.shape=} {proposed_X.shape=} {type(new_batch_indexes_batched)=} {len(new_batch_indexes_batched)=} {new_batch_indexes_batched=}")
                if self.task_type=='single_task_binary_classification': print(f"Average class before addition: {self.updated_full_y.mean()} (first one: 0.5090506076812744 ? - if pristine is initialized in the same way every time)")
                #if type(proposed_y)==torch.tensor: proposed_y=proposed_y.detach().cpu().numpy()
                final_proposed_X=proposed_X[new_batch_indexes_batched]
                final_proposed_y=proposed_y[new_batch_indexes_batched]
                self.updated_full_X=np.concatenate((self.updated_full_X,final_proposed_X),axis=0)
                self.updated_full_y=np.concatenate((self.updated_full_y,final_proposed_y),axis=0)
                #print(f"{ohe_to_seq(proposed_X[0])} {ohe_to_seq(proposed_X[1])} {ohe_to_seq(proposed_X[-1])}")
                if self.task_type=='single_task_binary_classification': print(f"Average class after addition: {self.updated_full_y.mean()} (for the new batch: {proposed_y[new_batch_indexes_batched].mean()}, full U: {proposed_y.mean()})") ## {proposed_y[randinds[new_batch_indexes_batched]].mean()})")
                if self.task_type=='single_task_binary_classification':
                    if math.isnan(self.updated_full_y.mean()): 
                        print("ERROR: average class is NaN")
                        exit()
                if self.task_type=='single_task_binary_classification': self.monitored_part2['averclass'].append(self.updated_full_y.mean())
                #print("-->",proposed_y[new_batch_indexes_batched])
                
            else: # NOT POOL BASED
                #print("CONTROL: 3")
                # Updata data module
                #print(f"{self.updated_full_X.shape=} {final_proposed_X.shape=} {type(new_batch_indexes_batched)=} {len(new_batch_indexes_batched)=} {new_batch_indexes_batched=}")
                if self.task_type=='single_task_binary_classification': print(f"Average class before addition: {self.updated_full_y.mean()} (first one: 0.5090506076812744 ? - if pristine is initialized in the same way every time)")
                #if type(final_proposed_y)==torch.tensor: final_proposed_y=final_proposed_y.detach().cpu().numpy()
                print(f"A2CHEKC {final_proposed_X.shape=}")
                self.updated_full_X=np.concatenate((self.updated_full_X,final_proposed_X),axis=0)
                self.updated_full_y=np.concatenate((self.updated_full_y,final_proposed_y),axis=0)
                #print(f"{ohe_to_seq(final_proposed_X[0])} {ohe_to_seq(final_proposed_X[1])} {ohe_to_seq(final_proposed_X[-1])}")
                if self.task_type=='single_task_binary_classification': print(f"Average class after addition: {self.updated_full_y.mean()} (for the new batch: {final_proposed_y.mean()}, full U: {final_proposed_y.mean()})") ## {final_proposed_y[randinds[new_batch_indexes_batched]].mean()})")
                if self.task_type=='single_task_binary_classification':
                    if math.isnan(self.updated_full_y.mean()): 
                        print("ERROR: average class is NaN")
                        exit()
                if self.task_type=='single_task_binary_classification': self.monitored_part2['averclass'].append(self.updated_full_y.mean())
                #print("-->",final_proposed_y[new_batch_indexes_batched])

            self.data_pruning()

            #print("CONTROL: 5")
            print("Making dal_dataset with new sequences.")
            #""
            dummy_data_module,dummy_h5_data_module_file=make_h5_data_module(final_proposed_X,final_proposed_y, 
                                                    self.data_module['X_test'],self.data_module['Y_test'],
                                                    self.data_module['X_test2'],self.data_module['Y_test2'],
                                                    self.data_module['X_valid'],self.data_module['Y_valid'],
                                                    batch_size=self.batch_size,
                                                    flag=str(self.seed_add)+'_proposed_iAL-'+str(self.current_i_AL), #str(self.seed_add+1000*self.model_first_index), 
                                                    outdir=self.outdir)
            #""
            self.data_module,self.h5_data_module_file=make_h5_data_module(self.updated_full_X,self.updated_full_y, #QUIQUIURG updated_full_X is ONLY the train, right????
                                                    self.data_module['X_test'],self.data_module['Y_test'],
                                                    self.data_module['X_test2'],self.data_module['Y_test2'],
                                                    self.data_module['X_valid'],self.data_module['Y_valid'],
                                                    batch_size=self.batch_size,
                                                    flag=str(self.seed_add), #str(self.seed_add+1000*self.model_first_index), # GOODOLD: str(self.seed_add), #str(self.seed_add)+'_iAL-'+str(self.current_i_AL),
                                                    outdir=self.outdir)
                ##self.h5_data_module_file=self.outdir+'dal_dataset_'+str(self.seed_add)+'.h5'
                                    
            #print(f"\n--- AL #{self.current_i_AL} finished: \nCurr. training X length: {len(self.updated_full_X)}, \nLast {str(list(self.monitored_part1.keys())[0])} monitored: {self.monitored_part1[list(self.monitored_part1.keys())[0]][-1]}, \nNew Indexes: (len: {len(new_batch_indexes_batched)})")
            if self.seq_method=='Xy-from-ds' or self.seq_method=='XdsYor':
                print(f"\n--- AL #{self.current_i_AL} finished: \nCurr. training X length: {len(self.updated_full_X)}, \nNew Indexes: (len: {len(new_batch_indexes_batched)})")
                os.system('date')
            else:
                print(f"\n--- AL #{self.current_i_AL} finished: \nCurr. training X length: {len(self.updated_full_X)}, \nNew Indexes: (len: {len(final_proposed_X)})")
                os.system('date')

            #print("CCC3")
            # Save results
            if self.current_i_AL % self.save_freq == 0: #QUIQUINONURG save_freq NEEDS to be 1, if the pipeline is split into two parts and used within Bash?
                print("Saving Metric numpy array at step",self.current_i_AL) #QUIQUINONURG Metric is a wrong word? As it is not a "distance"?
                #np.save(self.outdir+'cumpercuncs_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['cum_perc_uncs']))
                #np.save(self.outdir+'uncthrs_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2['unc_thrs']))
                #np.save(self.outdir+'already_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.already))
                np.save(self.outdir+'already_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.already))

                for key in self.monitored_part2.keys(): 
                    #np.save(self.outdir+'Res-'+key+'_'+self.outflag+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2[key]))
                    np.save(self.outdir+'Res-'+key+'_'+self.nickname+'_seedadd-'+str(self.seed_add)+'.npy',np.array(self.monitored_part2[key]))
        print("PART II COMPLETE.")         

    def Active_Learning_Loop(self):
        set_random_seed(self.seed_add)
        """ Perform Deep Active Learning loop """
        print("\n === === === ==== === Performing Deep Active Learning === === === === ===\n")
        for i_al in tqdm.tqdm(range(self.AL_cycles),total=self.AL_cycles,colour='red', desc='Active Learning Cycles'):           
            print()
            self.current_i_AL=i_al #QUIQUIURG should this be in Part_1? Or Part_2? Or both?
            self.Active_Learning_Cycle_Part_1()
            self.Active_Learning_Cycle_Part_2()
            #print(f"\n\n--- AL #{i_al} finished: \nCurr. training X length: {len(self.updated_full_X)}, \nPCC: {p_vals[0]}, \nNew Indexes: (len: {len(new_batch_indexes_batched)}) \nfirst ten: {new_batch_indexes_batched[:10]}, \nperc. of already: {float(len(already))/self.N_orig_train} ({len(already)} / {self.N_orig_train}) \n")
            #ctrlf.write(f"\n\n--- AL #{i_al} finished: \nCurr. training X length: {len(updated_full_X)}, \nPCC: {p_vals[0]}, \nNew Indexes: (len: {len(new_batch_indexes_batched)}) \nfirst ten: {new_batch_indexes_batched[:10]}, \nperc. of already: {float(len(already))/self.N_orig_train} ({len(already)} / {self.N_orig_train}) \n")
            
            print(f"After AL cycle: {torch.cuda.memory_allocated(device=device)=}") # https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html
            #torch.cuda.empty_cache() #QUIQUINONURG is this messing something up? # doesnt free anything
            #print(f"After emptying cache: {torch.cuda.memory_allocated(device=device)=}")
            #for j in range(self.N_Models): #QUIQUINONURG is this messing something up?
            #    self.DAL_Models[j].to('cpu') #QUIQUINONURG is this messing something up?
            #print(f"After freeing models: {torch.cuda.memory_allocated(device=device)=}")

        """
        if self.task_type=='single_task_regression':
            #return np.array(self.monitored_part1['PCC'])
            return {'PCC':np.array(self.monitored_part1['PCC']),'Spearman':np.array(self.monitored_part1['Spearman'])}
        elif 'classif' in self.task_type:
            #return np.array(self.monitored_part1['accuracy']),np.array(self.monitored_part1['AUROC']),np.array(self.monitored_part1['AUPR'])
            return {'accuracy':np.array(self.monitored_part1['accuracy']),'AUROC':np.array(self.monitored_part1['AUROC']),'AUPR':np.array(self.monitored_part1['AUPR'])}
        """
        #return {key:np.array(self.monitored_part1[key]) for key in self.monitored_part1.keys()} 
        return self.monitored_part1

##################################################################################################################

def get_outflag(args):
    # Large label for output, containing all relevant hyperparameters. #QUIQUINONURG remove outflag and use instead a dictionary within the dedicated subfolder of output
    # WARNING: ANY MODIFICATION WILL HAVE TO BE IMPLEMENTED EVEN IN DAL.SH
    outflag='Model-'+args.chosen_model+\
            '_DS-'+args.chosen_dataset+\
            '_pN-'+str(args.firstpristine)+\
            '_gU-'+str(args.generated_U)+\
            '_mxe-'+str(args.train_max_epochs)+\
            '_ALc-'+str(args.AL_cycles)+\
            '_itr-'+str(args.incremental_training)+\
            '_bao-'+str(args.how_many_new_batches_at_once)+\
            '_seqmeth-'+str(args.seq_method).replace('.','p')+'-mtrt-'+str(args.mutrate).replace('.','p')+\
            '_pristmeth-'+args.pristine_method+\
            '_unc-'+args.uncertainty_method+'-'+str(args.uncertainty_weight).replace('.','p')+\
            '_div-'+args.diversity_method+'-'+str(args.diversity_weight).replace('.','p')+\
            '_hpred-'+args.highpred_method+'-'+str(args.highpred_weight).replace('.','p')+\
            '_spdes-'+args.sp_desideratum
            #'_rank-'+str(args.rank_method) 
            #'_stop-'+str(args.perc_stop)+'-'+str(args.patience_AL)
            #'_unc-'+args.uncertainty_method+\ #QUIQUIURG ADD NEW ONES IF YOU ADDED NEW ARGS
            #QUIQUINONURG firstpristine makes pN redundant since it is contained in chosen_dataset already: can I use pN for the sequences per AL cycle? But that's covered by hmnbao and genU!
    if args.chosen_oracle=='PL_Oracle_Ensemble': outflag+='0deepens' #ORENS
    return outflag

def load_dataset_for_DAL(chosen_model,batch_size,seed=41,outdir='./'): #QUIQUINONURG is seed really necessary here?
    what_dataset='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    initial_ds_1=True
    #if args.initial_dataset=='default': #QUIQUINONURG not args
    #    what_dataset='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    #    initial_ds_1=True
    #else:
    #    what_dataset='./inputs/temp_'+args.initial_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    #    initial_ds_1=False
    #Model=Select_Model(chosen_model) 
    #Model=Select_Model(chosen_model=chosen_model, input_h5_file='"./inputs/'+args.chosen_dataset+'.h5"', initial_ds=True) #,input_h5_file='"./inputs/ATF2_200.h5"') #QUIQUINONURG args.input_file #QUIQUINONURG defining a model for nothing isnt great #QUIQUINONURG not args.
    #Model=Select_Model(chosen_model=chosen_model, input_h5_file='./inputs/'+args.chosen_dataset+'.h5', initial_ds=True) #GOODOLD #,input_h5_file='"./inputs/ATF2_200.h5"') #QUIQUINONURG args.input_file #QUIQUINONURG defining a model for nothing isnt great #QUIQUINONURG not args.
    Model=Select_Model(chosen_model=chosen_model, input_h5_file=what_dataset, initial_ds=initial_ds_1)
    X_train=Model.X_train
    y_train=Model.y_train
    X_test=Model.X_test
    y_test=Model.y_test
    X_valid=Model.X_valid
    y_valid=Model.y_valid
    N_orig_train=len(X_train)
    orig_data_module,_=make_h5_data_module(X_train,y_train,
                                    X_test,y_test,
                                    X_test,y_test,
                                    X_valid,y_valid,
                                    batch_size=Model.batch_size,
                                    flag='orig_seed-'+str(seed),
                                    outdir=outdir)
    return orig_data_module, N_orig_train, X_train,y_train, X_test,y_test, X_valid,y_valid

def make_pristine_dataset_for_model(chosen_model, Oracle_Model, #Oracle_trainer,
                                    pristine_N,pristine_method, #seq_method,
                                    inds,
                                    outdir,
                                    initial_ds,
                                    seedadd,
                                    i_AL
                                    ): ###PL V should be
    what_dataset='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    #if args.initial_dataset=='default': #QUIQUINONURG not args
    #    what_dataset='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    #else:
    #    what_dataset='./inputs/temp_'+args.initial_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5'
    #    initial_ds=False

    #Model=Select_Model(chosen_model=chosen_model, input_h5_file='"./inputs/'+args.chosen_dataset+'.h5"', initial_ds=initial_ds) #QUIQUINONURG defining a model for nothing isnt great #QUIQUINONURG not args.
    #Model=Select_Model(chosen_model=chosen_model, input_h5_file='./inputs/'+args.chosen_dataset+'.h5', initial_ds=initial_ds) #GOODOLD #QUIQUINONURG defining a model for nothing isnt great #QUIQUINONURG not args.
    #Model=Select_Model(chosen_model=chosen_model, input_h5_file='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5', initial_ds=initial_ds) #QUIQUINONURG "dummy_model"? #QUIQUINONURG actually, defining a model for nothing isnt great #QUIQUINONURG not args.  #supergroup
    Model=Select_Model(chosen_model=chosen_model, input_h5_file=what_dataset, initial_ds=initial_ds) #QUIQUINONURG "dummy_model"? #QUIQUINONURG actually, defining a model for nothing isnt great #QUIQUINONURG not args. 
    #X_pristine=X_train[inds] #QUIQUIURG This only if pristine_method=='random'
    
    # Pristine Xs
    #if seq_method=='Xy-from-ds' or seq_method=='XdsYor' or i_AL==0:
    if i_AL==0: #QUIQUIURG DEBUG I believe it is fine to get into che else with Xy-from-ds and with XdsYor when i_AL>0, but I should be certain of it
        print(f"DEBUG {type(Model.X_train)=} {inds=}")
        X_pristine=Model.X_train[inds] #QUIQUIURG This only if pristine_method=='random'
    else:
        X_pristine=h5py.File(outdir+'dal_dataset'+seedadd+'.h5','r')['X_train'] #QUIQUIURG is this fine for cases with seq_method neither Xy-from-ds nor XdsYor?
        #if type(X_pristine)==torch.tensor: X_pristine=X_pristine.detach().cpu()

    # Pristine Ys
    ##if seq_method=='Xy-from-ds': #QUIQUIURG seq_method should become pristine method maybe??? #supergroup
    if pristine_method=='ds':
        if i_AL==0:
            y_pristine=Model.y_train[inds] 
        else:
            y_pristine=h5py.File(outdir+'dal_dataset'+seedadd+'.h5','r')['Y_train']
        
        y_test_oracle=Oracle_Model.interrogate(Model.X_test).detach().cpu()
        y_valid_oracle=Oracle_Model.interrogate(Model.X_valid).detach().cpu()

        if Model.task_type=='single_task_binary_classification': #QUIQUINONURG not args
            y_pristine=y_pristine.round()
            y_test_oracle=y_test_oracle.round()
            y_valid_oracle=y_valid_oracle.round()

        """ #GOODOLD
        data_module,pristine_datafile=make_h5_data_module(X_pristine,y_pristine,
                                Model.X_test,Model.y_test,
                                Model.X_test,Model.y_test,
                                Model.X_valid,Model.y_valid,
                                batch_size=Model.batch_size,
                                flag='pristine_seed-'+str(args.jobseed)+'-'+str(args.model_index), #jobseed: needed so that every parallel run does not access or write the same file #QUIQUINONURG args -> input
                                outdir=outdir)
        """
        print(f"DEBUG PRISTINE: {Model.X_test.shape=} {y_test_oracle.shape=} {Model.X_test2,Model.y_test2.shape=} {Model.X_valid,Model.y_valid=}")
        data_module,pristine_datafile=make_h5_data_module(X_pristine,y_pristine,
                                Model.X_test,y_test_oracle,
                                Model.X_test2,Model.y_test2,
                                Model.X_valid,Model.y_valid,
                                batch_size=Model.batch_size,
                                flag='pristine_seed-'+str(args.jobseed)+'-'+str(args.model_index), #jobseed: needed so that every parallel run does not access or write the same file #QUIQUINONURG args -> input
                                outdir=outdir)
    #elif seq_method=='XdsYor':
    else: #Not ds, thus: probably never evoked #QUIQUIURG when making a distinction between seq_method and pristine_method this will have to change
    ##else: #QUIQUISOLVED? else should be correct in this specific line, right?
        
        #Oracle_trainer = pl.Trainer(max_epochs=Oracle_Model.train_max_epochs, logger=tb_logger) #, logger=None) #,patience=patience) #QUIQUINONURG is this better as an output of Acess_Model?
        #y_pristine=Oracle_trainer.predict(Oracle_Model, dataloaders=torch.utils.data.DataLoader(X_pristine, batch_size=model.batch_size, shuffle=True)) #infere_orcle(Oracle_Model,X_train[inds]) #QUIQUISOLVED predct_custom instead!!! anche se potrebbe essere molto, molto piu lento senza un batchsize! In realta predct_custom usa batch_size!!! Ma: solo per DeepSTARR: con InHouseCNN e gia fast enough.
        #y_pristine=Oracle_Model.predict_custom(X_pristine).detach().cpu() #goodold
        y_pristine=Oracle_Model.interrogate(X_pristine).detach().cpu()

        os.system('date')
        #y_test_oracle=Oracle_trainer.predict(Oracle_Model, dataloaders=torch.utils.data.DataLoader(Model.X_test, batch_size=model.batch_size, shuffle=True)) #infere_orcle(Oracle_Model,X_test)  #QUIQUISOLVED predct_custom instead!!! anche se potrebbe essere molto, molto piu lento senza un batchsize!  Ma: solo per DeepSTARR: con InHouseCNN e gia fast enough.
        #y_test_oracle=Oracle_Model.predict_custom(Model.X_test).detach().cpu() #interrogate
        y_test_oracle=Oracle_Model.interrogate(Model.X_test).detach().cpu()
        ##if seq_method=='XdsYor':
        ##    y_test2=Model.y_test          
        ##else:
        ##    y_test2=y_test_oracle #test2 becomes dummy when sequences are proposed
        y_test2=Model.y_test #QUIQUIURG is this exchanging test and test2 at every round???? # labels are ALWAYS possible for the test set (which is ALWAYS coming from the original dataset even when Sequence Proposals are used)

        os.system('date')
        #y_valid_oracle=Oracle_trainer.predict(Oracle_Model, dataloaders=torch.utils.data.DataLoader(Model.X_valid, batch_size=model.batch_size, shuffle=True)) #infere_orcle(Oracle_Model,X_valid)  #QUIQUISOLVED predct_custom instead!!! anche se potrebbe essere molto, molto piu lento senza un batchsize!  Ma: solo per DeepSTARR: con InHouseCNN e gia fast enough.
        #y_valid_oracle=Oracle_Model.predict_custom(Model.X_valid).detach().cpu() #goodold
        y_valid_oracle=Oracle_Model.interrogate(Model.X_valid).detach().cpu()

        if Model.task_type=='single_task_binary_classification': #QUIQUINONURG not args
            y_pristine=y_pristine.round()
            y_test_oracle=y_test_oracle.round()
            y_valid_oracle=y_valid_oracle.round()

        """ GOODOLD
        data_module,pristine_datafile=make_h5_data_module(X_pristine,y_pristine,
                                Model.X_test,y_test_oracle,
                                Model.X_test,y_test2,
                                Model.X_valid,y_valid_oracle,
                                batch_size=Model.batch_size,
                                flag='pristine_seed-'+str(args.jobseed)+'-'+str(args.model_index),
                                outdir=outdir)
        """
        data_module,pristine_datafile=make_h5_data_module(X_pristine,y_pristine,
                                Model.X_test,y_test_oracle,
                                Model.X_test2,y_test2,
                                Model.X_valid,y_valid_oracle,
                                batch_size=Model.batch_size,
                                flag='pristine_seed-'+str(args.jobseed)+'-'+str(args.model_index),
                                outdir=outdir)
#     else:
#         print("Wrongly selected sequence method.")
#         exit()        

    return data_module,pristine_datafile


################################################################################################


class PL_Oracle():
    def __init__(self, chosen_model, chosen_dataset, oracle_flag, initial_ds=True, extra_str=''): #this should be identical for every class oracle
        print(f"{oracle_flag=}")
        if oracle_flag=='same': oracle_flag=chosen_dataset 
        #ckptfile='oracle_'+chosen_model+'_'+chosen_dataset+'.ckpt' #these lines can be specific for the exact oracle #supergroup
        ckptfile='oracle_'+chosen_model+'_'+oracle_flag+'.ckpt' #these lines can be specific for the exact oracle
        input_h5_file='./inputs/'+chosen_dataset+'.h5' #QUIQUINONURG args.inpdir
        #model=eval('PL_'+chosen_model+'(input_h5_file="'+input_h5_file+'",initial_ds='+str(initial_ds)+')') #these lines can be specific for the exact oracle #supergroup
        """
        # Remove 16 June 2024
        extra_str=''
        if 'softplus' in ckptfile: extra_str=', activation="softplus_beta1"' #QUIQUINONURG may not be generalizable #TRUESOFT+
        """
        ##if 'softplus' in ckptfile: extra_str=", activation='softplus_beta1'" # SyntaxError: invalid syntax. Perhaps you forgot a comma?
        ##model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds='+str(initial_ds)+', extra_str='"+extra_str+"')") # QUIQUIURG doesnt change anything in terms of pred? it only counts the ckpt you use?? #QUIQUINOURG not args #goodold
        ##print("NOOOOOO",extra_str)
        model=eval("PL_"+chosen_model+"(input_h5_file='"+input_h5_file+"', initial_ds='+str(initial_ds)+', extra_str='"+extra_str+"')") # QUIQUIURG doesnt change anything in terms of pred? it only counts the ckpt you use?? #QUIQUINOURG not args
        ##print("NOOOOOO",model.has_aleatoric)
        if chosen_model=='NewResNet': print("NOOOOOO",model.model.unc_control)

        #model = model.load_from_checkpoint('./inputs/'+ckptfile, input_h5_file=input_h5_file) #QUIQUINONURG args.inpdir? #QUIQUIURG this SHOULD BE the most correct, but in principle it should not change anything because the test set remains the same all the time.
        model = model.load_from_checkpoint('./inputs/'+ckptfile, input_h5_file=input_h5_file, extra_str=extra_str) #QUIQUINONURG args.inpdir? #QUIQUIURG this SHOULD BE the most correct, but in principle it should not change anything because the test set remains the same all the time.
        self.model=model

    def interrogate(self, x):
        #return self.model.predict_custom(x)
        #return self.model.predict_custom(x.detach().cpu())
        return self.model.predict_custom(x.to(self.model.device))
    
class PL_Oracle_Ensemble():
    def __init__(self, chosen_model, chosen_dataset, oracle_flag, initial_ds=True, extra_str=''): #this should be identical for every class oracle
        print(f"{oracle_flag=}")
        if oracle_flag=='same': oracle_flag=chosen_dataset
        self.models=[]         
        for oracseed in [61,62,63,64,65]:
            ckptfile='oracle_'+chosen_model+'_'+oracle_flag+str(oracseed)+'.ckpt' #these lines can be specific for the exact oracle
            input_h5_file='./inputs/'+chosen_dataset+'.h5' #QUIQUINONURG args.inpdir
            model=eval("PL_"+chosen_model+"(input_h5_file='"+input_h5_file+"', initial_ds='+str(initial_ds)+', extra_str='"+extra_str+"')") # QUIQUIURG doesnt change anything in terms of pred? it only counts the ckpt you use?? #QUIQUINOURG not args
            if chosen_model=='NewResNet': print("NOOOOOO",model.model.unc_control)
            model = model.load_from_checkpoint('./inputs/'+ckptfile, input_h5_file=input_h5_file, extra_str=extra_str) #QUIQUINONURG args.inpdir? #QUIQUIURG this SHOULD BE the most correct, but in principle it should not change anything because the test set remains the same all the time.
            self.models.append(model)

    def interrogate(self, x):
        all_pred=[]
        for model in self.models:
            all_pred.append(model.predict_custom(x.to(self.models[0].device)))
        return torch.tensor(np.mean(all_pred,axis=0))
    
class EvoAug_DeepSTARR_Oracle():
    def __init__(self, chosen_dataset): #this should be identical for every class oracle
        ckptfile='DeepSTARR_finetune.ckpt' # DeepSTARR_aug.ckpt: 0.6988 (88 epochs), DeepSTARR_finetune.ckpt: 0.7062 # Download and train from: https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf
        input_h5_file='./inputs/'+chosen_dataset+'.h5' #QUIQUINONURG args.inpdir
        model=eval('PL_DeepSTARR(input_h5_file="'+input_h5_file+'",initial_ds=True)') #these lines can be specific for the exact oracle
        model = model.load_from_checkpoint('./inputs/'+ckptfile, input_h5_file=input_h5_file) #QUIQUINONURG args.inpdir? #QUIQUIURG this SHOULD BE the most correct, but in principle it should not change anything because the test set remains the same all the time.
        self.model=model

    def interrogate(self, x):
        return self.model.predict_custom(x.to(self.model.device))

def Select_Oracle(chosen_oracle='PL_Oracle', chosen_model='InHouseCNN', chosen_dataset='ATF2_200', oracle_flag='same', initial_ds=True, special_setting=None):
    if oracle_flag=='same': oracle_flag=chosen_dataset
    strings_list=[]
    if chosen_model!=None: strings_list.append('chosen_model="'+chosen_model+'"')
    if chosen_dataset!=None: strings_list.append('chosen_dataset="'+chosen_dataset+'"')
    if initial_ds!=None: strings_list.append('initial_ds="'+str(initial_ds)+'"')
    if oracle_flag!=None: strings_list.append('oracle_flag="'+oracle_flag+'"')
    if special_setting!=None: strings_list.append('extra_str="'+str(special_setting)+'"')
    strings=''
    for i_str,string in enumerate(strings_list):
        strings+=string
        if i_str!=len(strings_list)-1: strings+=', '
    print(chosen_oracle+'('+strings+')')
    Oracle_Model=eval(chosen_oracle+'('+strings+')')        
    return Oracle_Model


#################################################################################################


if __name__ == '__main__':

    #parser.add_argument('--', default=, type=)
    args = parser.parse_args()
    print(f"{args=}")

    overall_seed=41 #QUIQUI
    set_random_seed(overall_seed)

    outflag=get_outflag(args)
    print(f"args: {outflag=}")

    if not os.path.exists(args.outdir): os.system('mkdir '+args.outdir) #    os.makedirs(str(ABSPATH.parent) + '/data/meps/', exist_ok=True)
    #outpdir=args.outdir
    outpdir=args.outdir+outflag+'/' # WARNING: this will have to be addressed even in DAL.sh
    if not os.path.exists(outpdir): os.system('mkdir '+outpdir)
    if not os.path.exists(outpdir):
        print('ERROR: Failed to create output directory.')
        exit()

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"{device=}")

    print("\n=== === === LOAD DATASET")

    orig_data_module,N_orig_train,X_train,y_train,X_test,y_test,X_valid,y_valid=load_dataset_for_DAL(args.chosen_model,args.batch_size,seed=(args.jobseed+1000*args.model_index),outdir=outpdir)

    seed_add_4loading=str(args.jobseed+overall_seed+1000*args.model_index) #nAL and model_index can have the same power of 10! #1000*nAL+1*args.jobseed+overall_seed #str(self.seed_add)+'_iAL-'+str(self.current_i_AL)
    set_random_seed(int(seed_add_4loading)) # This is necessary for randomizing the pristine at i_AL=0 with jobseed
    # Select indexes at random within a certain range. "Already" contains the indexes (previously) picked, so it is both input and output
    #if args.initial_i_AL==0 and args.what_part=='one': #GOODOLD #QUIQUISOLVED? Is this dependent on what_part? Since on part_II you MUST have the same indexes
    if args.initial_i_AL==0 and args.what_part!='two': #QUIQUISOLVED? Is this dependent on what_part? Since on part_II you MUST have the same indexes
        inds,already=make_random_indexes_norepetitions(args.pristine_N,N_orig_train,already=[],save=True) #QUIQUISOLVED? inds and already are actually probably the same thing
        #print(f"{type(inds)=} {type(already)=} {inds==already}") #type(inds)=<class 'numpy.ndarray'> type(already)=<class 'list'> [ True  True  True ...  True  True  True]
        #np.save(outpdir+'already_'+outflag+'_seedadd-'+seed_add_4loading+'.npy',np.array(already))
        np.save(outpdir+'already_'+args.nickname+'_seedadd-'+seed_add_4loading+'.npy',np.array(already))
        print("File with indexes saved:",outpdir+'already_'+outflag+'_seedadd-'+seed_add_4loading+'.npy')
    else:
        #inds=list(np.load(outpdir+'already_'+outflag+'_seedadd-'+seed_add_4loading+'.npy')) #supergroup
        inds=list(np.load(outpdir+'already_'+args.nickname+'_seedadd-'+seed_add_4loading+'.npy')) #supergroup
        already=inds #QUIQUINONURG inds and already should actually be one and the same output in make_random_indexes_norepetitions
    inds=list(np.sort(inds)) #sort introduced with ResidualBind
    already=list(np.sort(already)) #sort introduced with ResidualBind

    if args.initial_i_AL==0:
        initial_ds=True #supergroup
        #if args.chosen_model!='ResidualBind':
        #    initial_ds=True
        #else:
        #    initial_ds=False
    else:
        initial_ds=False

    ######

    #QUIQUINONURG the oracle here should be evoked only if an Oracle is used somewhere (e.g if no method "y-from-DS")
    #torch.manual_seed(41) #41,42,43,44,45
    print("\n=== === === Model definition for Oracle")


    secondtest_oracle=False #QUIQUINONURG should be ok
    args_rank_method=args.uncertainty_method+'_'+str(args.uncertainty_weight)+'_'+args.diversity_method+'_'+str(args.diversity_weight)+'_'+args.highpred_method+'_'+str(args.highpred_weight) #QUIQUINONURG move elsewhere, and remove the horrible "args_"
    
    """
    ckpt_to_saveload='oracle_'+args.chosen_model+'_'+args.chosen_dataset+'.ckpt' 

    Oracle_Model,metrics_oracle=Access_Model(chosen_model=args.chosen_model,
                                                Train=False,
                                                model_file=ckpt_to_saveload,
                                                outdir=outpdir,
                                                #
                                                #batch_size=args.batch_size,train_max_epochs=args.train_max_epochs,patience=args.patience,
                                                batch_size=None,train_max_epochs=None,patience=None,min_delta=None,
                                                #data_module=None,
                                                #input_h5_file='"./inputs/'+args.chosen_dataset+'.h5"',
                                                #input_h5_file='./inputs/'+args.chosen_dataset+'.h5', #GOODOLD
                                                input_h5_file='./inputs/temp_'+args.chosen_dataset+'_'+args.nickname+'_mi-'+str(args.model_index)+'_js-'+str(args.jobseed)+'.h5',
                                                #input_h5_file=None, #goodold
                                                lr=None,
                                                initial_ds=True,
                                                flag='oracle_'+args_rank_method+'_js-'+str(args.jobseed)+'_mi-'+str(args.model_index),
                                                secondtest=secondtest_oracle) 
    print(f"Oracle Metrics: {metrics_oracle=}")
    """

    initial_ds_or=True #supergroup
    #if args.chosen_model!='ResidualBind':
    #    initial_ds_or=True
    #else:
    #    initial_ds_or=False
    if args.seq_method=='saliency_aleat': # or 'two_' in args.chosen_dataset: #QUIQUIURG or ok?
        Oracle_Model=Select_Oracle(args.chosen_oracle,args.chosen_model,args.chosen_dataset, oracle_flag=args.oracle_flag, initial_ds=initial_ds_or, special_setting='heteroscedastic') #QUIQUINONURG ckpt_to_saveload should become a passable argument(?)
    elif args.seq_method=='saliency_evidential':
        Oracle_Model=Select_Oracle(args.chosen_oracle,args.chosen_model,args.chosen_dataset, oracle_flag=args.oracle_flag, initial_ds=initial_ds_or, special_setting='evidential') #QUIQUINONURG ckpt_to_saveload should become a passable argument(?)
    else:
        Oracle_Model=Select_Oracle(args.chosen_oracle,args.chosen_model,args.chosen_dataset, oracle_flag=args.oracle_flag, initial_ds=initial_ds_or) #QUIQUINONURG ckpt_to_saveload should become a passable argument(?)

    #######
    print("\n=== === === Make Initial Dataset")

    if args.initial_i_AL==0: 
        print("Initial dataset will be the Pristine Dataset.")
        pristine_data_module,pristine_datafile=make_pristine_dataset_for_model(args.chosen_model, Oracle_Model, #Oracle_trainer,
                                        args.pristine_N,args.pristine_method, #args.seq_method,
                                        inds,
                                        outpdir,
                                        initial_ds,
                                        seed_add_4loading,
                                        args.initial_i_AL) 

        data_module=pristine_data_module
        data_file=pristine_datafile
    else:
        print("Initial dataset will be the one saved at Part I of the previous AL cycle.")
        #print("Loading:",outpdir+'dal_dataset_'+seed_add_4loading+'_iAL-'+str(args.initial_i_AL-1)+'.h5')
        #data_module=h5py.File(outpdir+'dal_dataset_'+seed_add_4loading+'_iAL-'+str(args.initial_i_AL-1)+'.h5','r') 
        print("Loading:",outpdir+'dal_dataset_'+seed_add_4loading+'.h5') #+'_iAL-'+str(args.initial_i_AL-1)+'.h5')
        lock_till_found(outpdir+'dal_dataset_'+seed_add_4loading+'.h5')
        data_file=outpdir+'dal_dataset_'+seed_add_4loading+'.h5'
        data_module=h5py.File(data_file,'r') #+'_iAL-'+str(args.initial_i_AL-1)+'.h5','r') 
        dummy=data_module['X_train'].shape #this line is only useful to check if data_module was correctly loaded and that its containing file was not corrupted. #QUIQUINONURG remove when ready

    print("\n=== === === Deep Active Learning")

    for nAL in tqdm.tqdm(range(args.N_Active_Learning),total=args.N_Active_Learning,desc='Loop over DALs',colour='cyan'):
        seed_add_assign=100000*nAL+1*args.jobseed+overall_seed+1000*args.model_index
        #print(f"{seed_add_assign=} {seed_add_4loading=} {seed_add_assign==seed_add_4loading=}")
        if str(seed_add_assign)!=str(seed_add_4loading): 
            print("ERROR: Seed misassignment.")
            exit()
        #print(f"{seed_add_assign=} {seed_add_4loading=}") #QUIQUINONURG these names should be made more meaningful, and probably one variable is redundant
        print(" ::::::::::::::::::::::::::::::::::::::::::::::::::::: New DAL Cycle:",nAL)
        os.system('date')
        print(f"QQQQ {data_file=}")
        DAL=Deep_Active_Learning_Cycles(initial_i_AL=args.initial_i_AL, model_first_index=args.model_index,
                                        chosen_model=args.chosen_model, #task_type=task_type, 
                                        Oracle_Model=Oracle_Model, 
                                        data_module=data_module, orig_data_module=orig_data_module, data_file=data_file,
                                        #X_valid=X_valid, y_valid=y_valid, 
                                        outflag=outflag, 
                                        nickname=args.nickname,
                                        seed_add=seed_add_assign, 
                                        outdir=outpdir,
                                        batch_size=args.batch_size, 
                                        already=already, 
                                        save_freq=args.save_freq,
                                        AL_cycles=args.AL_cycles, 
                                        N_Models=args.N_Models, 
                                        train_max_epochs=args.train_max_epochs, 
                                        learning_rate=args.learning_rate, 
                                        patience=args.patience,
                                        min_delta=args.min_delta,
                                        incremental_training=args.incremental_training, 
                                        sigmadistr_freq=args.sigmadistr_freq,
                                        seq_method=args.seq_method, 
                                        #rank_method=args.rank_method, 
                                        uncertainty_method=args.uncertainty_method,diversity_method=args.diversity_method,highpred_method=args.highpred_method,
                                        uncertainty_weight=args.uncertainty_weight,diversity_weight=args.diversity_weight,highpred_weight=args.highpred_weight,
                                        sp_desideratum=args.sp_desideratum,
                                        mutrate=args.mutrate,
                                        generated_U=args.generated_U, 
                                        how_many_new_batches_at_once=args.how_many_new_batches_at_once)
        print("DAL object defined")
        os.system('date')
        if args.what_part=='all': 
            learning_metric=DAL.Active_Learning_Loop()
        elif args.what_part=='one': 
            DAL.Active_Learning_Cycle_Part_1()
            #print("DEBUG: will skip Part 1")
        elif args.what_part=='two': 
            DAL.Active_Learning_Cycle_Part_2()
        #for key in learning_metrics.keys():
        #    learning_metrics[key].append(learning_metric[key])
        DAL.reset_for_new_AL_cycle() #QUIQUINONURG questo per evitare di nuovo out of memory(???)

    print("SOLVE ALL QUIQUIs")

    print("\noutflag: "+outflag+"\n")
    print("SCRIPT END.")
    #ctrlf.close()

"""

# https://wandb.ai/acrnjar
# https://docs.wandb.ai/quickstart

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

"""


"""
From Chandana
# ==============================================================================
# Imports
# ==============================================================================
# pip install wandb
import wandb
from wandb.keras import WandbCallback

# ==============================================================================
# W&B logging
# ==============================================================================
# say you have some config dictionary of parameters you want to log
wandb_project = config['wandb_project'] # give a project name here
wandb_name = config['wandb_name'] # give a run/trial name here

wandb.init(project=wandb_project, name=wandb_name, config=config)

# ==============================================================================
# Load dataset, build model, something like
# ==============================================================================
model = model_zoo.base_model(**model_config)

model.compile(
    tf.keras.optimizers.Adam(lr=config['lr']),
    loss='mse',
    metrics=[Spearman, pearson_r] # these metrics will be logged using wandb callback
    )

# ==============================================================================
# train model using wandb callback
# ==============================================================================
history = model.fit(
            x_train, y_train,
            epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            shuffle=True,
            validation_data=(x_valid, y_valid),
            callbacks=[WandbCallback(save_model=(False))] # set to true if you want
            )

# ==============================================================================
# Evaluate model and log performance on test set
# ==============================================================================
mse, pcc, scc = summary_statistics(model, x_test,  y_test)

wandb.log({
    'MSE': mse,
    'PCC':  pcc,
    'SCC':  scc,
})

"""