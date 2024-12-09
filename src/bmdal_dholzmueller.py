#https://github.com/dholzmueller/bmdal_reg/blob/main/examples/using_bmdal.ipynb

#pip install bmdal_reg
# from bmdal_reg.bmdal.feature_data import TensorFeatureData
# from bmdal_reg.bmdal.algorithms import select_batch
# from bmdal_reg.bmdal.algorithms import BatchSelectorImpl, MaxDetSelectionMethod #AC
import os
import sys
# sys.path.append('../../')
# from bmdal_reg_AC import *
# from bmdal_reg_AC.bmdal_reg.bmdal.feature_data import TensorFeatureData
# from bmdal_reg_AC.bmdal_reg.bmdal.algorithms import select_batch
# from bmdal_reg_AC.bmdal_reg.bmdal.algorithms import BatchSelectorImpl, MaxDetSelectionMethod #AC
#sys.path.append('../../bmdal_reg_custom')
import tqdm
import is_seq_in_xtrain

if os.uname()[1]=='auros': 
    sys.path.append('/home/alessandro/Documents/GitHub_Local/bmdal_reg_custom/')
elif os.uname()[1]=='amethyst': 
    sys.path.append('/home/acrnjar/Desktop/TEMP/GitHub_Local/bmdal_reg_custom/')
else:
    sys.path.append('/grid/koo/home/crnjar/bmdal_reg_custom/')
"""
feature_data.py
super().__init__(n_samples=data.shape[0], device=data.device, dtype=data.dtype)   ### -2 -> 0
"""
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch
from bmdal_reg.bmdal.algorithms import BatchSelectorImpl, MaxDetSelectionMethod #AC
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import is_seq_in_xtrain

def batch_selection_method(
    x_train, x_pool, n_to_make, models, y_train,
    selection_method='maxdet',device='cuda',external_batch_size=100,
    base_kernel='grad', kernel_transforms=[('rp', [512])],
    sel_with_train=False,
    ):
    #selection_method='maxdet' #  BatchBALD: MaxDet-P
    #selection_method='bait'
    #selection_method='kmeanspp' #  BADGE: KMeansPP-P

    """
    :param precomp_batch_size: Batch size used for precomputations of the features.
    :param nn_batch_size: Batch size used for passing the data through the NN.
    #AC this choice does not affect the result for new_idxs
    """
    #precomp_batch_size=32768
    #nn_batch_size=8192
    #precomp_batch_size=n_to_make #AC this choice does not affect the result for new_idxs
    #nn_batch_size=n_to_make #AC this choice does not affect the result for new_idxs
    precomp_batch_size=external_batch_size #QUIQUIURG NOT USED!!!
    nn_batch_size=external_batch_size

    #sel_with_train=False #sel_with_train: TP/P-mode: BatchBALD should be MaxDet-P

    # train_data = TensorFeatureData(x_train)
    # pool_data = TensorFeatureData(x_pool)
    #train_data = TensorFeatureData(torch.tensor(x_train).to(device)) #WORKS FOR RESIDUALBIND
    #pool_data = TensorFeatureData(torch.tensor(x_pool).to(device)) #WORKS FOR RESIDUALBIND
    train_data = TensorFeatureData(torch.tensor(x_train))
    pool_data = TensorFeatureData(torch.tensor(x_pool))

    #print(f"{dir(train_data)=}")
    #print(f"{vars(train_data)=}") #vars(train_data)={'n_samples': 4, 'device': device(type='cuda', index=0), 'dtype': torch.float32, 'data': tensor([[[2.8979e-02, 4.0190e-01, 2.5
    #print(f"{train_data.data.shape=}")
    #print(f"{train_data.n_samples=}")
    #exit()

    new_idxs, results_dict = select_batch(batch_size=n_to_make, models=models, 
                            data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                            selection_method=selection_method, sel_with_train=sel_with_train,
                            #base_kernel='grad', kernel_transforms=[('rp', [512])]) # return batch_idxs, results_dict : def select within class BatchSelectorImpl in algorithms.py
                            base_kernel=base_kernel, kernel_transforms=kernel_transforms) # return batch_idxs, results_dict : def select within class BatchSelectorImpl in algorithms.py
    print(f"BMDALDEBUG {len(x_train)=} {len(x_pool)=} {len(new_idxs)=}")
    #return new_idxs.detach().cpu() #WORKS FOR RESIDUALBIND
    return new_idxs

if __name__=='__main__':
    import set_torch_tensors_test

    np.random.seed(1234) # AC
    torch.manual_seed(1234)

    # n_train = 100
    # n_pool = 2000

    # n_train = 20000
    # n_pool = 100000
    # n_to_select=n_train

    # n_train = 5000
    # n_pool = 25000
    # n_to_select=n_train

    # n_train = 20000
    # n_pool = 100000
    # n_to_select=n_train

    # n_train = 2000
    # n_pool = 10000
    # n_to_select=n_train

    # n_train = 100
    # n_pool = 10000

    batch_size=100

    #case='default'
    #case='K562'
    #case='RBFOX1'
    case='DeepSTARR'

    #for n_train in [100,1000,10000,20000]:
    #for n_train in [20000,10000]:
    for n_train in [20000]:
    #for n_train in [2000]:
        #for selection_method in ['kmeanspp','maxdet','lcmd','bait']: #/home/alessandro/Documents/GitHub_Local/bmdal_reg_custom/bmdal_reg/bmdal/algorithms.py
        for selection_method in ['lcmd']:
            #for n_pool in [100,1000,10000,20000,100000]:
            for n_pool in [100000]:
            #for n_pool in [10000]:
                for setreduce in ['no']: #,'yes']:
                    for mutrate in [0.05]: #0.25,0.05]:
                        print(f"========= for fill: {selection_method=} {n_train=} {n_pool=} {setreduce=} {mutrate=}")
                        n_to_select=n_train

                        if case=='default':
                            x = torch.randn(n_train+n_pool, 3)
                            theta = 3*(x[:, 1] + 0.1 * x[:, 0])
                            x = (0.2 * x[:, 2] + x[:, 1] + 2)[:, None] * torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
                            y = torch.exp(x[:, 0])
                            y = y[:, None]
                            x_train = x[:n_train]
                            y_train = y[:n_train]
                            x_pool = x[n_train:]
                            y_pool = y[n_train:]
                            custom_model = nn.Sequential(nn.Linear(2, 100), nn.SiLU(), nn.Linear(100, 100), nn.SiLU(), nn.Linear(100, 1))
                        if case=='K562' or case=='RBFOX1' or case=='DeepSTARR':
                            from PL_Models import *
                            import quick_proposer
                            #x_train=torch.rand((n_train,4,230))
                            #x_train=torch.rand((n_train,4,41))
                            #y_train=torch.rand((n_train,1))
                            #x_pool=torch.rand((n_pool,4,230))
                            #x_pool=torch.rand((n_pool,4,41))
                            #y_pool=torch.rand((n_pool,1))
                            if case=='K562':
                                chosen_model='LegNetPK'
                                #chosen_model='ResidualBind'
                                inputdir='../inputs/'
                                chosen_dataset='newLentiMPRAK562_labels-seed0_random0_20000'
                                #chosen_dataset='newLentiMPRAHepG2_labels-seed0_random0_20000'
                                #chosen_dataset='RBFOX1_rnacompete2013labels-seed0_random0_20000'
                            if case=='RBFOX1':
                                chosen_model='ResidualBind'
                                inputdir='../inputs/'
                                chosen_dataset='RBFOX1_rnacompete2013labels-seed0_random0_20000'
                            if case=='DeepSTARR':
                                chosen_model='DeepSTARR'
                                inputdir='../inputs/'
                                chosen_dataset='DeepSTARRdev_labels-seed0_random0_20000'

                            device='cpu'
                            #device='cuda' #comment for NODEVICE

                            data=h5py.File(inputdir+chosen_dataset+'.h5','r')
                            x_train=torch.tensor(np.array(data['X_train']))[:n_train].to(device)
                            y_train=torch.tensor(np.array(data['Y_train']))[:n_train].to(device)
                            # x_train=torch.tensor(np.array(data['X_train']))[:n_train] #NODEVICE
                            # y_train=torch.tensor(np.array(data['Y_train']))[:n_train] #NODEVICE
                            #x_pool=torch.tensor(np.array(data['X_test']))

                            # import experiment_hairpin
                            # x_pool=torch.empty(0)
                            # for _ in tqdm.tqdm(range(n_pool)):
                            #     x=random_ohe_seq(sequence_length)
                            #     print(x.shape)
                            #     exit()
                            #     x_pool=torch.cat((x_pool,x),axis=0)

                            extra_str=''
                            custom_model=eval("PL_"+chosen_model+"(input_h5_file='"+inputdir+chosen_dataset+".h5', initial_ds=True, extra_str='"+extra_str+"')")
                            m_idx=61
                            if 'K562' in chosen_dataset: ckptfile='../inputs/oracle_'+chosen_model+'_newLentiMPRAK562_processed_for_dal_finetune.ckpt'                   
                            if 'RBFOX1' in chosen_dataset: ckptfile='../inputs/oracle_ResidualBind_RBFOX1_rnacompete2013_processed_for_dal_relustandard1.ckpt'
                            if 'RBFOX1' in chosen_dataset: ckptfile='../inputs/oracle_ResidualBind_RBFOX1_rnacompete2013_processed_for_dal_relustandard1.ckpt'
                            if 'DeepSTARR' in chosen_dataset: ckptfile='../inputs/oracle_DeepSTARR_DeepSTARRdev_finetune61.ckpt'
                            # if 'K562' in chosen_dataset: ckptfile='../inputs/oracle_'+chosen_model+'_newLentiMPRAK562_processed_for_dal_finetune'+str(m_idx)+'.ckpt'                   
                            # if 'HepG2' in chosen_dataset: ckptfile='../inputs/oracle_'+chosen_model+'_newLentiMPRAHepG2_processed_for_dal_finetune'+str(m_idx)+'.ckpt'                   
                            # if 'RBFOX1' in chosen_dataset: ckptfile='../inputs/oracle_ResidualBind_RBFOX1_rnacompete2013_processed_for_dal_relustandard1'+str(m_idx)+'.ckpt'                   
                            custom_model = custom_model.load_from_checkpoint(ckptfile, input_h5_file=inputdir+chosen_dataset+'.h5')

                            x_pool=quick_proposer.quick_proposer_func(X_source=torch.tensor(np.array(data['X_test'])),seq_method='mutation',n_to_make=n_pool,chosen_model=chosen_model,Models=[custom_model.to(device)],batch_size=batch_size,mutant_fraction=mutrate).to(device)
                            #x_pool=quick_proposer.quick_proposer(X_source=torch.tensor(np.array(data['X_test'])),seq_method='totally_random',n_to_make=n_pool,chosen_model=chosen_model,Models=[custom_model.to(device)],batch_size=batch_size).to(device)
                            #x_pool=quick_proposer.quick_proposer(X_source=torch.tensor(np.array(data['X_test'])),seq_method='mutation',n_to_make=n_pool,chosen_model=chosen_model,Models=[custom_model],batch_size=batch_size,mutant_fraction=0.25)  #NODEVICE

                            #print(f"{custom_model.predict_custom(x_pool[:2])=}")
                            #print(f"{custom_model.predict_custom(x_pool[:2].to(device))=}")

                            #x_pool=is_seq_in_xtrain.contained_method_3(x_pool,x_train)
                            #proposed_X=torch.tensor(np.array(set_torch_tensors_test.lookup_dna_to_ohe(list(set(set_torch_tensors_test.lookup_ohe_to_dna(proposed_X))-set(set_torch_tensors_test.lookup_ohe_to_dna(self.updated_full_X)))))) #QUIQUIURG this doesnt actually work for tensors!
                            if setreduce=='yes': 
                                new_x_pool=torch.tensor(np.array(set_torch_tensors_test.lookup_dna_to_ohe(list(set(set_torch_tensors_test.lookup_ohe_to_dna(x_pool))-set(set_torch_tensors_test.lookup_ohe_to_dna(x_train))))),dtype=torch.float32) 
                                if new_x_pool.shape!=x_pool.shape: print(f"fill: setreduce was applied: {new_x_pool.shape=} {x_pool.shape=}")
                                x_pool=new_x_pool

                            #exit()
                        

                        print(f"{x_train.shape=}")
                        print(f"{y_train.shape=}")
                        print(f"{x_pool.shape=}")
                        #print(f"{y_pool.shape=}")

                        # new_x_pool=is_seq_in_xtrain.contained_method_2(x_pool,x_train)
                        # print(f"{len(new_x_pool)=} {len(x_pool)=}")

                        if case=='default':
                            opt = torch.optim.Adam(custom_model.parameters(), lr=2e-2)
                            for epoch in range(256):
                                y_pred = custom_model(x_train)
                                loss = ((y_pred - y_train)**2).mean()
                                train_rmse = loss.sqrt().item()
                                pool_rmse = ((custom_model(x_pool) - y_pool)**2).mean().sqrt().item()
                                #print(f'train RMSE: {train_rmse:5.3f}, pool RMSE: {pool_rmse:5.3f}')
                                loss.backward()
                                opt.step()
                                opt.zero_grad()

                        #models=[custom_model.to('cuda')] #works, pre 16 oct 2024
                        models=[custom_model.to(device)]
                        #models=[custom_model] # NODEVICE
                        #selection_method='maxdet' #  BatchBALD: MaxDet-P #generates issues of filling
                        #selection_method='bait'
                        #selection_method='kmeanspp' #  BADGE: KMeansPP-P

                        # for kernel_transforms in [
                        #                         ('grad',[('rp', [512])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('train', [1000])]), #DOESNT WORK
                        #                         ('grad',[('train', [1000000])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('pool', [1000])]), #DOESNT WORK
                        #                         ('grad',[('pool', [1000000])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('train', [1000000])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('train', [1000000]),('pool', [1000000])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('acs-grad', [1000])]), #DOESNT WORK
                        #                         ('grad',[('rp', [512]),('acs-grad', [100000])]), #DOESNT WORK
                        #                             ]:
                        for base_kernel in ['grad']: #'ll' #    'll', 'grad', 'lin', 'nngp', 'ntk', and 'laplace'.
                            # "lin": not found?
                            # "nngp":   File "/grid/koo/home/crnjar/bmdal_reg_custom/bmdal_reg/bmdal/feature_maps.py", line 263, in sketch      raise NotImplementedError()
                            # 'ntk':   File "/grid/koo/home/crnjar/bmdal_reg_custom/bmdal_reg/bmdal/feature_maps.py", line 263, in sketch    raise NotImplementedError()
                            # 'laplace':   File "/grid/koo/home/crnjar/bmdal_reg_custom/bmdal_reg/bmdal/feature_maps.py", line 263, in sketch    raise NotImplementedError()
                            for kernel_transforms in [
                                                [('rp', [512])], 
                                                # [('rp', [512]),('train', [1000])], #DOESNT WORK
                                                # [('train', [1000000])], #DOESNT WORK
                                                # [('rp', [512]),('pool', [1000])], #DOESNT WORK
                                                # [('pool', [1000000])], #DOESNT WORK
                                                # [('rp', [512]),('train', [1000000])], #DOESNT WORK
                                                # [('rp', [512]),('train', [1000000]),('pool', [1000000])], #DOESNT WORK
                                                # [('rp', [512]),('acs-grad', [1000])], #DOESNT WORK
                                                # [('rp', [512]),('acs-grad', [100000])], #DOESNT WORK
                                                # [('rp', [512]),('train', [1000000]),('pool', [1000000])], 
                                                # [('rp', [512]),('train', [2000000]),('pool', [2000000])], 
                                                # [('rp', [512]),('train', [5000000]),('pool', [5000000])], 
                                                # [('rp', [512]),('train', [10000000]),('pool', [10000000])], 
                                                    ]:
                                if selection_method=='lcmd':
                                    sel_with_train=True
                                else:
                                    sel_with_train=False
                                # /home/acrnjar/Desktop/TEMP/GitHub_Local/bmdal_reg_custom/bmdal_reg/bmdal/algorithms.py

                                #./bmdal/selection.py:                self.status = f'filling up with random samples because selection failed after n_selected = {len(self.selected_idxs)}'
                                #./bmdal/selection.py:                self.status = f'removing the latest overselected samples because the backward step failed '\
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512])])
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512]),('train', [1000000])])
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512]),('pool', [1000000])])
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512]),('train', [1000000]),('pool', [1000000])])
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512]),('acs-grad', [1000])])
                                #final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel='grad', kernel_transforms=[('rp', [512]),('acs-grad', [100000])])
                                final_idxs=batch_selection_method(x_train, x_pool,n_to_select, models, y_train, selection_method, base_kernel=base_kernel, kernel_transforms=kernel_transforms, sel_with_train=sel_with_train)
                                print("for fill:",base_kernel,kernel_transforms)
                                print(f"{final_idxs=}")
                                print("fill: ___________") # to space them out
                                torch.cuda.empty_cache()

                        if case=='default':
                            myfigsize=(12,10)
                            fig = plt.figure(1, figsize=myfigsize)
                            # plt.plot(x_pool[:, 0].numpy(), x_pool[:, 1].numpy(), '.', color='grey')
                            # plt.plot(x_train[:, 0].numpy(), x_train[:, 1].numpy(), '.', color='black')
                            # plt.plot(x_pool[final_idxs, 0].numpy(), x_pool[final_idxs, 1].numpy(), '.', color='red')
                            plt.scatter(x_pool[:, 0].numpy(), x_pool[:, 1].numpy(), color='grey',s=50) 
                            plt.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), color='black',s=100, alpha=0.5) 
                            plt.scatter(x_pool[final_idxs, 0].numpy(), x_pool[final_idxs, 1].numpy(), color='red', s=100, alpha=0.5)
                            fig.savefig(selection_method+'.png',dpi=600, bbox_inches='tight') 
                            #plt.show()

                        #print(f"{results_dict=}") #results_dict={'kernel_time': {'total': 0.011530354000569787, 'process': 0.09254582699999947}, 'selection_time': {'total': 0.024850796999089653, 'process': 0.19264199100000035}, 'selection_status': None}
    print("SCRIPT END")

"""
TRAIL:
algorithms.py
elif selection_method == 'maxdet':
alg = MaxDetFeatureSpaceSelectionMethod
later:
batch_idxs = alg.select(batch_size)
so that finally: 
return batch_idxs, results_dict

selection.py: 
class MaxDetFeatureSpaceSelectionMethod(IterativeSelectionMethod):

sel_with_train: TP/P-mode: BatchBALD should be MaxDet-P
"""

"""
try to find a "minimum failing example" - like, try to simplify everything until the bug goes away? For example:
- use input features ("linear kernel")
- try to see if the error prevails when you use MaxDist
- then try to copy out the MaxDist code but replace the distance calculation with a straightforward distance calculation using norm(a-b).
"""