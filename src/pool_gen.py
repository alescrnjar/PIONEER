import torch
import numpy as np
import pandas as pd
from math import ceil
from collections import deque
from time import time
import pickle 
import tqdm

import h5py
import os
import dinuc_shuf_AC

import PL_Models_interpr_utils

if os.uname()[1]=='galaxy1':
    import sys
    sys.path.append('../D3-DNA-Discrete-Diffusion/Training_and_Sampling_Conditioned')
    import sampling
    import load_model
    from torch.utils.data import TensorDataset

class SequenceProposer(object):
    def __init__(self, generation_method, sequence_length, seed=None, 
                 track_time=True, track_uncertanties = True, track_batches=False, track_hamming=True, track_pref = './',
                 mutable_window =  (None, None), upstream = '', downstream='',
                 source_method='random'):

        self.generation_method = generation_method
        self.sequence_length = sequence_length
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        
        self.track_time = track_time
        self.track_uncertanties = track_uncertanties
        self.track_batches = track_batches
        self.track_hamming = track_hamming
        self.track_pref = track_pref
        
        if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
            self.tracker = {}
        
        self.mutable_window =  mutable_window
        self.upstream = upstream
        self.downstream=downstream

        self.source_method=source_method
        
        if generation_method == 'totally_random':
            self.generate_batch = self.random_sampler
        elif generation_method == 'mutation':
            self.generate_batch = self.mutate_randomly
        elif generation_method == 'realmut':
            self.generate_batch = self.mutate_randomly
        elif generation_method == 'BatchBALDsubsel':
            self.generate_batch = self.mutate_randomly
        elif generation_method == 'BADGEsubsel':
            self.generate_batch = self.mutate_randomly
        elif generation_method == 'BADGEfromt':
            self.generate_batch = self.random_sampler
        elif generation_method == 'LCMDsubsel':
            self.generate_batch = self.mutate_randomly
        elif generation_method == 'LCMDfromt':
            self.generate_batch = self.random_sampler
        elif generation_method == 'concatBADGE1':
            print("concatBADGE1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatLCMD1':
            print("concatLCMD1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatrand1':
            print("concatrand1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatTEMPALCMD1':
            print("concatLCMD1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatTEMPArand1':
            print("concatrand1 doesnt have a generation method, will pass.")

        elif generation_method == 'Costmixrand1':
            print("Costmixrand1 doesnt have a generation method, will pass.")            
        elif generation_method == 'CostmixLCMD1':
            print("Costmixrand1 doesnt have a generation method, will pass.")     
        elif generation_method == 'CostmixTEMPArand1':
            print("Costmixrand1 doesnt have a generation method, will pass.")            
        elif generation_method == 'CostmixTEMPALCMD1':
            print("Costmixrand1 doesnt have a generation method, will pass.")     
        elif generation_method == 'PriceHundredLCMD':
            print("Costmixrand1 doesnt have a generation method, will pass.")   
        elif generation_method == 'Price20KLCMD':
            print("Costmixrand1 doesnt have a generation method, will pass.")  

        elif generation_method == 'saliency':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'realsal':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'realTEMPAsal':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'LCMDfromJ':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'saliency_y':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'saliency_aleat':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'saliency_evidential':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'saliency_div_y':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'GradientSHAP':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'DeepLiftSHAP':
            self.generate_batch =self.mutate_by_saliency
        elif generation_method == 'saliency_U-A':
            self.generate_batch =self.mutate_by_saliency
        
        elif generation_method == 'simulated_annealing':
            self.generate_batch = self.simulated_annealing
        elif generation_method == 'simulated_annealing_y':
            self.generate_batch = self.simulated_annealing
        
        elif generation_method == 'greedy':
            self.generate_batch = self.greedy
        elif generation_method == 'greedy_y':
            self.generate_batch = self.greedy

        elif generation_method == 'genetic':
            self.generate_batch = self.genetic
        elif generation_method == 'genetic_y':
            self.generate_batch = self.genetic
        
        elif generation_method=='evoaug': #AC
            self.generate_batch = self.evoaug
        elif generation_method=='evoaugmut': #AC
            self.generate_batch = self.evoaug
        elif generation_method=='evoaugassign': #AC
            self.generate_batch = self.evoaug
        elif generation_method=='realevoaug': #AC
            self.generate_batch = self.original_evoaug

        elif generation_method == 'dinuc': #AC
            self.generate_batch = self.dinuc
        elif generation_method == 'LCMDfromd':
            self.generate_batch = self.dinuc


        elif generation_method == 'fromfile': #AC
            self.generate_batch = self.select_from_file
        elif generation_method == 'vanilla_diffusion': #AC
            self.generate_batch = self.vanilla_diffusion
        elif generation_method == 'diffusion_file': #AC
            self.generate_batch = self.diffusion_file
        elif generation_method == 'diffusion_y': #AC
            self.generate_batch = self.diffusion_y
        else:
            assert False, "%s has not been implemented as a sequence generation method"
            


    def dinuc(self, x_source, n_to_make, cycles=1, batch_size=100): #AC
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        originals=torch.tensor(originals)
        X_new=torch.empty(0)
        for x in originals:
            xp=torch.tensor(x).unsqueeze(0).permute(0,2,1)
            x_din=dinuc_shuf_AC.dinuc_shuffle_AC(xp[0])
            X_new=torch.cat((X_new,x_din.unsqueeze(0).permute(0,2,1)),axis=0)
        return X_new

    def get_hamming(self, seqs):
        match = seqs[None,::] == seqs[:,None,::] # batch x batch x 4 x L array with True/False entries
        matching_nucs = match.all(axis=-2) # batch x batch x L : whether or not the nt matches
        hamm_pairs = matching_nucs.sum(axis=-1) # sum over length: i.e. number of matches

        fraction_match = hamm_pairs/seqs.shape[-1] # turns into a percentage (batch x batch)
        return(fraction_match)
    
    def sample_seq_distribution(self, n_to_make, distribution):
        return(torch.Tensor(self.rng.multinomial(1, distribution, size=(n_to_make,self.sequence_length) ).transpose((0,2,1))))
    
    def random_sampler(self, n_to_make,ranker=None):
        if self.track_time:
            start_time = time()
        
        distribution = [1/4.0]*4
        output = self.sample_seq_distribution(n_to_make, distribution)
        
        if self.track_time:
            end_time = time()
            self.tracker['time'] = [end_time-start_time]
            
        if self.track_uncertanties:
            #des = ranker.calculate_desiderata(output).detach().cpu().numpy()
            des = ranker.calculate_desiderata(output)[0].detach().cpu().numpy()
            self.tracker['ave_unc'] = [des.mean()]
            self.tracker['std_unc'] = [des.std()]
            
        if self.track_batches:
            self.tracker['batch'] = [output.detach().cpu().numpy()]
        
        if self.track_hamming:
            self.tracker['ham'] = [self.get_hamming(output.detach().cpu().numpy())]
            
        if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
            with open(self.track_pref+'.pckl', 'wb') as f:
                pickle.dump(self.tracker, f)
            
        return(output)
    
    #def select_from_source(self, n_to_make, x_source, replace = False, method='random', ranker=None):
    def select_from_source(self, n_to_make, x_source, replace = False, ranker=None):
        method=self.source_method
        if method=='random':
            idx = self.rng.choice(x_source.shape[0], n_to_make, replace=replace)
            selected=x_source[np.sort(idx)]
        elif method=='pristine':
            selected=x_source[:n_to_make] #QUIQUIURG I think this should be fine
        elif method=='highest':
            preds=ranker.Predictive_Models[0].predict_custom(x_source) #QUIQUIURG should not just be [0]
            idx=(np.argsort(preds)[::-1])[:n_to_make]
            selected=x_source[np.sort(idx)]
        return selected
    

    def select_from_file(self, 
                         chosen_model, file_to_open, n_to_make, ALcycle=0, always_from_zero=False,
                         path_to_alr_inds='./fromfile_alrinds.npy', path_to_alr_inds_AL0='./fromfile_alrinds.npy'): #, replace = False, ranker=None):
        
        data = h5py.File(file_to_open, 'r')
        
        if chosen_model=='ResidualBind':
            X_imported=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
        elif chosen_model=='DeepSTARR':
            X_imported = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
        elif chosen_model=='LegNetPK':
            X_imported=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
        elif chosen_model=='NewResNet':
            X_imported=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
        else:
            print("ERROR in select_from_file: wrong chosen model")
            exit()

        if not always_from_zero: #path will contain the USABLE indeces in this case
            if ALcycle==0:
                idxs=list(np.load(path_to_alr_inds_AL0))
            else:
                idxs=list(np.load(path_to_alr_inds))
        else:  #path will contain the ALREADY USED indeces in this case
            idxs=np.load(path_to_alr_inds_AL0)
            if ALcycle!=0:
                giavisti=list(np.load(path_to_alr_inds)) 
                i_to_keep=np.isin(idxs,giavisti,assume_unique=True,invert=True) #test_indexes_rank_fromfile.py
                idxs=list(idxs[i_to_keep])

        n_to_pick=n_to_make
        print(f"FROMFILE: {ALcycle=} {n_to_pick=} {len(idxs)=}")
        if n_to_pick>len(idxs): 
            n_to_pick=len(idxs)
        proposed_idxs=np.random.choice(idxs,size=n_to_pick,replace=False)
        new_X=torch.tensor(X_imported[proposed_idxs])

        if not always_from_zero: #in the other case, will be saved from within DAL_Pipeline
            for ii in proposed_idxs:
                idxs.remove(ii)
            np.save(path_to_alr_inds,np.array(idxs))
        print(f"FROMFILE: just before returning: {len(idxs)=}")
        return new_X

    def make_mutants(self,originals,mutation_numbers):
        n_to_make = len(originals)
        to_mutate = np.array(originals)
        positions = np.array([self.rng.choice(self.sequence_length,size  = mutation_numbers, replace =  False) for i in range(n_to_make)])
        seqs, _ = np.indices(positions.shape)
        
        pvals = ((1- to_mutate[seqs.flatten(),:, positions.flatten()]).astype('float64'))
        for i_nt in range(pvals.shape[0]): #QUIQUINONURG this is not optimal probably.
            pvals[i_nt,:]/=pvals.sum(axis=1)[i_nt]
        mutations = self.rng.multinomial(1, pvals, size=(n_to_make*mutation_numbers) )
        to_mutate[seqs.flatten(),:, positions.flatten()] = mutations        
        return(to_mutate)
    
    def mutate_randomly(self, n_to_make, x_source, mutation_number, cycles=1, ranker=None):
        if self.track_time:
            start_time = time()
        
        to_mutate_final=torch.empty(0) #NTOMAKELARGER
        nrounds=int(n_to_make/len(x_source)) #NTOMAKELARGER
        n2make=len(x_source)
        if nrounds==0: 
            nrounds=1
            n2make=n_to_make
        for iround in range(nrounds): #NTOMAKELARGER

            originals = self.select_from_source(n2make, x_source, replace = False) #NTOMAKELARGER
            if self.track_uncertanties:            
                des = ranker.calculate_desiderata(originals)[0].detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp

            if self.track_batches:

                tmp = self.tracker.get('batch', [])
                tmp.append(originals)
                self.tracker['batch'] = tmp

            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(originals))
                self.tracker['ham'] = tmp
                
            to_mutate = np.array(originals)
            
            for c in range(cycles):
                to_mutate = self.make_mutants(to_mutate,mutation_number)
            
                if self.track_time:
                    end_time = time()
                    tmp = self.tracker.get('time', [])
                    tmp.append(end_time-start_time)
                    self.tracker['time'] = tmp

                if self.track_uncertanties:
                    des = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()

                    tmp = self.tracker.get('ave_unc', [])
                    tmp.append(des.mean())
                    self.tracker['ave_unc'] = tmp

                    tmp = self.tracker.get('std_unc', [])
                    tmp.append(des.std())
                    self.tracker['std_unc'] = tmp
                    

                if self.track_batches:
                    
                    tmp = self.tracker.get('batch', [])
                    tmp.append(to_mutate)
                    self.tracker['batch'] = tmp

                if self.track_hamming:
                    tmp = self.tracker.get('ham', [])
                    tmp.append(self.get_hamming(to_mutate))
                    self.tracker['ham'] = tmp
                    
                if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                    with open(self.track_pref+'.pckl', 'wb') as f:
                        pickle.dump(self.tracker, f)
                
            to_mutate_final=torch.cat((to_mutate_final,torch.Tensor(to_mutate)),axis=0) #NTOMAKELARGER

        return to_mutate_final #NTOMAKELARGER
    
    def saliency_one_step(self, to_optimize, mutation_numbers, ranker, temp): #, to_backprop='unc'):
        
        # Copy array
        to_mutate = np.array(to_optimize)
        
        seqs, nts, pos =  to_mutate.shape
        batch_sails = []
        [m.zero_grad() for m in ranker.Predictive_Models]

        #if 'saliency' in self.generation_method:
        if 'saliency' in self.generation_method or \
           'concatBADGE1' in self.generation_method or \
           'concatLCMD1' in self.generation_method or \
           'concatrand1' in self.generation_method or \
           'Costmixrand1' in self.generation_method or \
           'CostmixLCMD1' in self.generation_method or \
           'realsal' in self.generation_method or \
           'concatTEMPABADGE1' in self.generation_method or \
           'concatTEMPALCMD1' in self.generation_method or \
           'concatTEMPArand1' in self.generation_method or \
           'CostmixTEMPArand1' in self.generation_method or \
           'CostmixTEMPALCMD1' in self.generation_method or \
           'PriceHundredLCMD' in self.generation_method or \
           'Price20KLCMD' in self.generation_method or \
           'LCMDfromJ' in self.generation_method or \
           'realTEMPAsal' in self.generation_method:

            print(f"DEBUG4DSRR:{to_mutate.shape=} {ceil(seqs/ranker.batch_size)=} {seqs=} {ranker.batch_size=}")
            for batch in np.split(to_mutate, ceil(seqs/ranker.batch_size)): #np.array_split: for uneven split?
                x = torch.tensor(batch).float().requires_grad_()
                x.retain_grad()

                if self.generation_method=='saliency_aleat':
                    preds_av = ranker.get_pred(x, keep_grads = True)
                elif self.generation_method=='saliency_evidential':
                    preds_av, aleat_uncs, epist_uncs = ranker.pred_unc_evidential(x, keep_grads = True)
                else:
                    unc_all, preds_av = ranker.calculate_desiderata(x, keep_grads = True)
                if self.generation_method=='saliency':
                    unc_all.mean().backward()
                elif self.generation_method=='realsal':
                    unc_all.mean().backward()
                elif self.generation_method=='concatBADGE1':
                    unc_all.mean().backward()
                elif self.generation_method=='concatLCMD1':
                    unc_all.mean().backward()
                elif self.generation_method=='concatrand1':
                    unc_all.mean().backward()
                elif self.generation_method=='Costmixrand1':
                    unc_all.mean().backward()
                elif self.generation_method=='CostmixLCMD1':
                    unc_all.mean().backward()
                elif self.generation_method=='realTEMPAsal':
                    unc_all.mean().backward()
                elif self.generation_method=='LCMDfromJ':
                    unc_all.mean().backward()
                elif self.generation_method=='concatTEMPABADGE1':
                    unc_all.mean().backward()
                elif self.generation_method=='concatTEMPALCMD1':
                    unc_all.mean().backward()
                elif self.generation_method=='concatTEMPArand1':
                    unc_all.mean().backward()
                elif self.generation_method=='CostmixTEMPArand1':
                    unc_all.mean().backward()
                elif self.generation_method=='CostmixTEMPALCMD1':
                    unc_all.mean().backward()
                elif self.generation_method=='PriceHundredLCMD':
                    unc_all.mean().backward()
                elif self.generation_method=='Price20KLCMD':
                    unc_all.mean().backward()

                #elif to_backprop=='y':
                elif self.generation_method=='saliency_y':
                    preds_av.mean().backward()
                elif self.generation_method=='saliency_aleat':
                    print(f"ALEATDEBUG: {preds_av.shape=}")
                    preds_av[:,1].mean().backward()
                elif self.generation_method=='saliency_div_y':
                    #preds_av.std().mul(-1).backward() #pre 21 feb 2024
                    preds_av.std().backward()
                elif self.generation_method=='saliency_U-A':
                    unc_all.mean().add(preds_av.mean().multiply(-1)).backward()
                elif self.generation_method=='saliency_evidential':
                    #preds_av.std().mul(-1).backward() #pre 21 feb 2024
                    epist_uncs.std().backward()
                else:
                    print("ERROR: to_backprop incorrectly set.")
                    exit()
                
                unc_sail = x.grad.data.cpu().numpy() #QUIQUINONURG rename to "sail"
                #print('unc_sail: ',unc_sail)
                batch_sails.append(unc_sail)
                [m.zero_grad() for m in ranker.Predictive_Models]

        elif 'GradientSHAP' in self.generation_method:
            for bbatch in np.array_split(to_mutate, ceil(len(to_mutate)/10)):
                grad=PL_Models_interpr_utils.captum_gradientshap_no_y_no_index(bbatch,model=ranker.Predictive_Models[0],todevice=True, apply_corr=True, null_method='standard', device='cuda', num_background = 1000)
                batch_sails.append(grad.detach().cpu().numpy())

        elif 'DeepLiftSHAP' in self.generation_method:
            for index in range(len(to_mutate)):
                X_attr=deep_lift_shap(ranker.Predictive_Models[0], torch.tensor(to_mutate), target=index, random_state=self.seed)
                batch_sails.append(X_attr)

        unc_sail = np.concatenate(batch_sails)
        
        if temp == 'neg_inf':
            
            # make WT saliencies minimal to stop selection
            cut = np.array(unc_sail)
            cut[to_mutate.astype(bool)] = cut.min()-1 # in case of N: all four options remain viable
            
            # identify the maximal nucleotide and position combos
            max_nts = cut.argmax(axis=1, keepdims=False) # in case of N: ok because N is never the maximum anyway (because it's not even an option)
            i, j = np.indices((cut.shape[0],cut.shape[2]))
            parted_poses = cut[i,max_nts, j].argpartition(-mutation_numbers,axis=1,)
            
            #mutate the array
            to_mutate[i[:,-mutation_numbers:], :, parted_poses[:,-mutation_numbers:]] = 0
            to_mutate[i[:,-mutation_numbers:], max_nts[i,parted_poses][:,-mutation_numbers:], parted_poses[:,-mutation_numbers:]] = 1

            # in case of N: basically over cycles you will get rid of all Ns
                        
        elif isinstance(temp, float) or isinstance(temp, int):
            unc_sail = unc_sail.astype('float64')
            old_seq_mask = to_mutate.astype(bool)
            unc_sail[old_seq_mask] = 0 #QUIQUI stoping old sequences from messing up the scaling should probably fix this
            
            #QUIQUI This is a hacky solution to stop low numbers from all becoming zero in the np.exp should probably fix this
            
            #QUIQUI This is a hacky solution to stop overflow errors in the np.exp should probably fix this
            scale_thresh = np.log(1e+40)*temp            
            mask = (unc_sail>=scale_thresh).any(axis=-1).any(axis=-1)
            maxes = unc_sail[mask].max(axis=-1).max(axis=-1)
            unc_sail[mask] = unc_sail[mask]-(maxes-scale_thresh)[:,None,None] 
                 
            unc_sail[old_seq_mask] = -np.inf #QUIQUI setting old sequence probabilities to zero this is prob fine
            ex = np.exp(unc_sail/temp)
            ex = ex + 1e-10
            ex[old_seq_mask] = 0
            probs = (ex/ex.sum(axis=1).sum(axis=1)[:,None,None])
            badmask = np.isnan(probs).any(axis=-1).any(axis=-1)
            if badmask.sum()!=0:
                print(f"NANCHECK: {np.isnan(ex).sum()=} (>0 if contains at least one nan) {np.isinf(ex).sum()=} (>0 if contains at least one inf)")

            pos_to_mut = np.stack([self.rng.choice(np.arange(probs.shape[2]), p=probs[i].sum(axis=0), size = mutation_numbers,replace=False) for i in range(probs.shape[0])])
            i, j = np.indices(pos_to_mut.shape)

            pos_to_mut = pos_to_mut.flatten()
            i = i.flatten()

            to_mutate[i, :, pos_to_mut] = self.rng.multinomial(1, probs[i, :, pos_to_mut]/probs[i, :, pos_to_mut].sum(axis=1)[:,None])

        else:
            assert False, "invalid temp"
        
        return(to_mutate)
        
    def mutate_by_saliency(self, n_to_make, x_source, cycles, mutations_per, ranker, temp, decay=None): #, to_backprop='unc'):
        if self.track_time:
            start_time = time()
        
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        if self.track_uncertanties:
            des = ranker.calculate_desiderata(originals)[0].detach().cpu().numpy()

            tmp = self.tracker.get('ave_unc', [])
            tmp.append(des.mean())
            self.tracker['ave_unc'] = tmp

            tmp = self.tracker.get('std_unc', [])
            tmp.append(des.std())
            self.tracker['std_unc'] = tmp

        if self.track_batches:

            tmp = self.tracker.get('batch', [])
            tmp.append(originals)
            self.tracker['batch'] = tmp

        if self.track_hamming:
            tmp = self.tracker.get('ham', [])
            tmp.append(self.get_hamming(originals))
            self.tracker['ham'] = tmp

        to_mutate = np.array(originals)
        
        for c in range(cycles):
            if (decay is None) or (temp == 'neg_inf'):
                t = temp
            elif cycles>1:
                t = decay + (temp-decay)*((((cycles-1)-c)/(cycles-1))**2) # calculate temperature for accepting mutations
            else:
                t = decay
            
            to_mutate = self.saliency_one_step(to_mutate, mutations_per, ranker, t) #, to_backprop=to_backprop)

            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp

            if self.track_uncertanties:
                des = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp

            if self.track_batches:
                
                tmp = self.tracker.get('batch', [])
                tmp.append(to_mutate)
                self.tracker['batch'] = tmp

            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(to_mutate))
                self.tracker['ham'] = tmp

            if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                with open(self.track_pref+'.pckl', 'wb') as f:
                    pickle.dump(self.tracker, f)

        return(torch.Tensor(to_mutate))    




    def gen_batch_idxs(self, num, batch_size, max_batches=1000):
        d = deque(np.arange(num))
        for i in range(max_batches):
            l = []
            for i in range(batch_size):
                l.append(d[0])
                d.rotate(-1)
            yield np.array(l)
        print("WARNING: max batch number exeeded. No sequence was accepted (given the current temperature) that resulted in an increase of the desideratum. This particular cycle will result in the original unchanged sequence.")
        yield None
    
    def simulated_annealing(self, n_to_make, x_source, batch_size, mutations_per_cycle, mutation_cycles, temp, decay,
                            ranker, min_frac=0, min_abs= 0, prevent_decreases = True, max_batches=100): #max_batches=10):

        if self.track_time:
            start_time = time()
            
        originals = self.select_from_source(n_to_make, x_source, replace = False) #Selected sequences
        if self.track_uncertanties:
            desid,preds_av = ranker.calculate_desiderata(originals).detach().cpu().numpy()
            if self.generation_method=='simulated_annealing':
                des=desid
            elif self.generation_method=='simulated_annealing_y':
                des=preds_av

            tmp = self.tracker.get('ave_unc', [])
            tmp.append(des.mean())
            self.tracker['ave_unc'] = tmp

            tmp = self.tracker.get('std_unc', [])
            tmp.append(des.std())
            self.tracker['std_unc'] = tmp
                

        if self.track_batches:

            tmp = self.tracker.get('batch', [])
            tmp.append(originals)
            self.tracker['batch'] = tmp

        if self.track_hamming:
            tmp = self.tracker.get('ham', [])
            tmp.append(self.get_hamming(originals))
            self.tracker['ham'] = tmp
            
        to_mutate = np.array(originals) #Selected sequences copied to mutate
        
        uncs = []
        # Calculate initial uncertainty so we know what we need to improve over
        uncs = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()
        
        # cycle over number of mutations
        for cycle in range(mutation_cycles):
            if mutation_cycles>1:
                t = decay + (temp-decay)*((((mutation_cycles-1)-cycle)/(mutation_cycles-1))**2) # calculate temperature for accepting mutations
            else:
                t = decay
            unmuted = np.arange(to_mutate.shape[0]) # maintain a list of indices in to_mutate that need to be mutated
            batches = self.gen_batch_idxs(len(unmuted), batch_size, max_batches=max_batches) # produces indexes in unmuted that refer to indecies in to_mutate as batches
            l = 0 
            while len(unmuted)>0:
                l+=1
                unmuted_idxs = next(batches) # get the next batch of indexes in unmuted that refer to indecies in to_mutate to mutate
                if unmuted_idxs is None:
                    break
                batch_idxs = unmuted[unmuted_idxs] # get batch indecies with reference to to_mutate
                batch_unmut = to_mutate[batch_idxs] # get sequences to mutate from to_mutate
                batch_mut = self.make_mutants(batch_unmut,mutations_per_cycle) # mutate sequences
                unc_old = uncs[batch_idxs] # get uncertainties of unmutated sequences in batch order with indicies with reference to to_mutate
                
                # get new mutant uncertainties
                x = torch.tensor(batch_mut).float()#.to(device)
                unc_new = ranker.calculate_desiderata(x)[0].detach().cpu().numpy()
                
                
                difference = (1+min_frac)*unc_old - unc_new + min_abs # get differences between old and new uncs
            
                # Get acceptance probabilities in batch order
                acceptance_prob = (difference<0).astype(float)
                bad_mask = np.logical_not(acceptance_prob)
                acceptance_prob[bad_mask] = np.exp(-1*difference[bad_mask]/t)
                if prevent_decreases:
                    acceptance_prob[unc_new<unc_old] = 0

                pulls = self.rng.uniform(size= acceptance_prob.shape) # get random vartiates in batch order

                accept = acceptance_prob>pulls # get accepted batch elements in batch order
                if sum(accept)>0:
                    accepted_batch_idxs = batch_idxs[accept] #get indecies in to_mutate that belong to accepted sequences in accepted order
                    unc_new_accepted = unc_new[accept] #get unc_new values that belong to accepted sequences in accepted order
                    unmuted_idxs_accepted = unmuted_idxs[accept] # get indecies in unmuted of accepted sequences in accepted order

                    sort_mask = np.argsort(unc_new_accepted)[::-1] # Get sort indecies for accepted sequences in descending order
                    sort_unc_new_accepted = unc_new_accepted[sort_mask] # sort the unc_new sequences in descending order
                    sort_accepted_batch_idxs = accepted_batch_idxs[sort_mask]  # get indecies in to_mutate that belong to accepted sequences in descending unc order
                    sort_unmuted_idxs_accepted = unmuted_idxs_accepted[sort_mask]  # get indecies in unmuted that belong to accepted sequences in descending unc order
                    
                    # uniques_accepted_batch_idxs is deduplicated indicies in to_mutate for accepted sequences
                    # max_idxs is indecies into the accpeted sequences in descending unc order for the highest value of each unique to_mutate index
                    uniques_accepted_batch_idxs, max_idxs = np.unique(sort_accepted_batch_idxs, return_index=True)
                    
                    unmuted = np.delete(unmuted, sort_unmuted_idxs_accepted[max_idxs]) # remove accepted values from unmuted
                    batches = self.gen_batch_idxs(len(unmuted), batch_size, max_batches=max_batches) # Generate a new batch generator to generate indecies in the new unmuted
        
                    # index uncs with to_mutate indecies and assign sorted new uncs indexed for maximum vunique values
                    uncs[uniques_accepted_batch_idxs] = sort_unc_new_accepted[max_idxs] 
                
                    # index to_mutate with to_mutate indecies and assign with batch mutants while striping to accepted sorting and cutting to max values
                    to_mutate[uniques_accepted_batch_idxs] = batch_mut[accept][sort_mask][max_idxs] 
                    
            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp

            if self.track_uncertanties:
                #des = ranker.calculate_desiderata(to_mutate).detach().cpu().numpy()
                des = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp

            if self.track_batches:
                
                tmp = self.tracker.get('batch', [])
                tmp.append(to_mutate)
                self.tracker['batch'] = tmp

            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(to_mutate))
                self.tracker['ham'] = tmp
                
            if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                with open(self.track_pref+'.pckl', 'wb') as f:
                    pickle.dump(self.tracker, f)
                

        return(torch.Tensor(to_mutate))
        
    def gen_all_singles_batches(self, to_optimize, batch_size, window):
        seqs, nts, posses = to_optimize.shape
        to_hit = np.logical_not(to_optimize[:,:,window[0]:window[1]])
        seq, nt, pos = np.where(to_hit)
        i = 0
        for s_batch, n_batch, p_batch in zip(np.array_split(seq, ceil(len(seq)/batch_size)), 
                                             np.array_split(nt, ceil(len(nt)/batch_size)), 
                                             np.array_split(pos, ceil(len(pos)/batch_size))):
            batch = to_optimize[s_batch]
            b_idx = np.arange(len(s_batch))
            if window[0] is None:
                offset = 0
            else:
                offset =window[0]
            batch[b_idx, :, p_batch+offset] = 0
            batch[b_idx, n_batch, p_batch+offset] = 1

            eq = np.not_equal(to_optimize[s_batch], batch)
            #assert all(eq.reshape(-1,nts, posses).sum(axis=-1).sum(axis=-1)==2), "Mutation numbers are wrong:\n%s"%str(eq.reshape(-1,nts, posses).sum(axis=-1).sum(axis=-1))
            yield(batch)

    def gen_singles_by_idx(self, to_optimize, idxs, window):
        seqs, nts, posses = to_optimize.shape
        #to_hit = np.logical_not(to_optimize[:,:,window[0]:window[1]])
        to_hit=np.ones(to_optimize[:,:,window[0]:window[1]].shape).astype(bool)
        seq, nt, pos = np.where(to_hit)
        i = 0
        batch = to_optimize[seq[idxs]]
        b_idx = np.arange(len(seq[idxs]))
        if window[0] is None:
            offset = 0
        else:
            offset =window[0]
                
        batch[b_idx, :, pos[idxs]+offset] = 0
        batch[b_idx, nt[idxs], pos[idxs]+offset] = 1

        eq = np.not_equal(to_optimize[seq[idxs]], batch)
        #assert all(eq.reshape(-1,nts, posses).sum(axis=-1).sum(axis=-1)==2), "Mutation numbers are wrong:\n%s"%str(eq.reshape(-1,nts, posses).sum(axis=-1).sum(axis=-1))
        return(batch)
                
    def greedy(self, n_to_make, x_source, batch_size, mutation_number, ranker):
        if self.track_time:
            start_time = time()
            
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        print(f"greedy: {originals.shape=}")
        if self.track_uncertanties:
            if self.generation_method=='greedy_y':
                des = ranker.calculate_desiderata(originals)[1].detach().cpu().numpy()
            else:
                des = ranker.calculate_desiderata(originals)[0].detach().cpu().numpy()

            tmp = self.tracker.get('ave_unc', [])
            tmp.append(des.mean())
            self.tracker['ave_unc'] = tmp

            tmp = self.tracker.get('std_unc', [])
            tmp.append(des.std())
            self.tracker['std_unc'] = tmp

        if self.track_batches:

            tmp = self.tracker.get('batch', [])
            tmp.append(originals)
            self.tracker['batch'] = tmp

        if self.track_hamming:
            tmp = self.tracker.get('ham', [])
            tmp.append(self.get_hamming(originals))
            self.tracker['ham'] = tmp
            
        to_mutate = np.array(originals)
        seqs, nts, pos =  to_mutate.shape
        #copies = (nts-1)*pos # case of no Ns
        copies = nts*pos
        new_array=np.ones((seqs, nts, pos))*-99999.
        
        for mut in range(mutation_number):
            print('working on mut: ',mut)
            batch_uncs = []
            for batch in self.gen_all_singles_batches(to_mutate, ranker.batch_size, 
                                                      self.mutable_window):
                x = torch.tensor(batch).float()
                if self.generation_method=='greedy_y':
                    batch_uncs.append(ranker.calculate_desiderata(x)[1].detach().cpu().numpy())
                else:
                    batch_uncs.append(ranker.calculate_desiderata(x)[0].detach().cpu().numpy())
            unc_all = np.concatenate(batch_uncs)
            new_array[np.where(np.logical_not(to_mutate))]=unc_all
            
            unc_all=new_array

            unc_all = unc_all.reshape((seqs,copies))
            max_unc_idx = np.argmax(unc_all, axis=-1)
            rec_idx = np.ravel_multi_index((np.arange(seqs),max_unc_idx), (seqs,copies))
            to_mutate = self.gen_singles_by_idx(to_mutate, rec_idx,  self.mutable_window)
            
            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp

            if self.track_uncertanties:
                des = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp

            if self.track_batches:
                
                tmp = self.tracker.get('batch', [])
                tmp.append(to_mutate)
                self.tracker['batch'] = tmp

            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(to_mutate))
                self.tracker['ham'] = tmp
                
            if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                with open(self.track_pref+'.pckl', 'wb') as f:
                    pickle.dump(self.tracker, f)
            
        return(torch.Tensor(to_mutate))
    
    def genetic(self, n_to_make, x_source, mutation_number, cycles, x_prob, expansion_fold, ranker):
        if self.track_time:
            start_time = time()
            
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        if self.track_uncertanties:
            des = ranker.calculate_desiderata(originals)[0].detach().cpu().numpy()

            tmp = self.tracker.get('ave_unc', [])
            tmp.append(des.mean())
            self.tracker['ave_unc'] = tmp

            tmp = self.tracker.get('std_unc', [])
            tmp.append(des.std())
            self.tracker['std_unc'] = tmp

        if self.track_batches:

            tmp = self.tracker.get('batch', [])
            tmp.append(originals)
            self.tracker['batch'] = tmp

        if self.track_hamming:
            tmp = self.tracker.get('ham', [])
            tmp.append(self.get_hamming(originals))
            self.tracker['ham'] = tmp
            
        to_mutate = np.array(originals)
        seq_len = to_mutate.shape[-1]
        nts = to_mutate.shape[-2]
        
        for cycle in range(cycles):
        
            # select crossover events
            pvals = np.zeros((n_to_make, n_to_make))
            self_prob = 1-x_prob
            pvals[:,:] = x_prob/(n_to_make-1)
            pvals[np.diag_indices(n_to_make)] = self_prob
            second_seq = np.stack([self.rng.choice(np.arange(n_to_make),size=expansion_fold, p=pval) for pval in pvals]).T

            #select crossover positions
            position = np.stack([self.rng.choice(np.arange(seq_len),size=expansion_fold) for i in range(n_to_make)]).T
            position_bool_map = np.arange(seq_len)[None,None, :]<=position[:,:,None]

            #convert to indecies First seq
            expansion_idx, first_seq_idx, position_idx = np.where(position_bool_map)

            #generate crossovers
            new_mutants = np.zeros((expansion_fold, n_to_make, nts, seq_len))
            new_mutants[expansion_idx,first_seq_idx, :, position_idx] = to_mutate[first_seq_idx,:,position_idx]
            
            #convert to indecies second seq
            expansion_idx, first_seq_idx, position_idx = np.where(np.logical_not(position_bool_map))
            second_seq_idx  = second_seq[expansion_idx,first_seq_idx]
            
            new_mutants[expansion_idx,first_seq_idx, :, position_idx] = to_mutate[second_seq_idx,:,position_idx]
            
            new_mutants = new_mutants.reshape((expansion_fold*n_to_make, nts, seq_len))
            
            # Introduce point mutants
            new_mutants = torch.Tensor(self.make_mutants(new_mutants,mutation_number)).float()
            
            batch_uncs = []
            desid_0,preds_av_0 = ranker.calculate_desiderata(new_mutants) #.detach().cpu().numpy()
            if self.generation_method=='genetic':
                desid_1=desid_0.detach().cpu().numpy()
            elif self.generation_method=='genetic_y':
                desid_1=preds_av_0.detach().cpu().numpy()
            unc_all=desid_1 #QUIQUINONURG this dummy name will better be replaces

            to_mutate =  new_mutants[np.argpartition(unc_all, -n_to_make)[-n_to_make:]]
            
            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp

            if self.track_uncertanties:
                des = ranker.calculate_desiderata(to_mutate)[0].detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp

            if self.track_batches:
                
                tmp = self.tracker.get('batch', [])
                tmp.append(to_mutate)
                self.tracker['batch'] = tmp

            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(to_mutate))
                self.tracker['ham'] = tmp
                
            if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                with open(self.track_pref+'.pckl', 'wb') as f:
                    pickle.dump(self.tracker, f)
        return(to_mutate)
        
    def batch_from_file(self):
        assert False, "Not implemented"