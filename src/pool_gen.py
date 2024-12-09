import torch
import numpy as np
import pandas as pd
from math import ceil
from collections import deque
from time import time
import pickle 
import tqdm

import potts_AC
import hessian_AC

import augment
import evoaug_augmentations

import h5py
import os
import dinuc_shuf_AC

import PL_Models_interpr_utils

import gen_synth_embed

#from tangermeme.ersatz import insert
#from tangermeme.utils import one_hot_encode   # Convert a sequence into a one-hot encoding
#from tangermeme.utils import characters  
#from tangermeme.deep_lift_shap import deep_lift_shap

#if os.path.isdir('../D3-DNA-Discrete-Diffusion/'):
if os.uname()[1]=='galaxy1':
    import sys
    #sys.path.append('../D3-DNA-Discrete-Diffusion/Training and Sampling')
    sys.path.append('../D3-DNA-Discrete-Diffusion/Training_and_Sampling_Conditioned')
    import sampling
    #import load_model_local
    import load_model
    from torch.utils.data import TensorDataset

class SequenceProposer(object):
    def __init__(self, generation_method, sequence_length, seed=None, 
                 track_time=True, track_uncertanties = True, track_batches=False, track_hamming=True, track_pref = './',
                 mutable_window =  (None, None), upstream = '', downstream='',
                 source_method='random'):
        """
        TODO: standardize generate_batch inputs
        TODO: Add sequence contexts
        TODO: Add comments
        TODO: Add help text for each function
        """
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
            #self.generate_batch = self.mutate_randomly
            print("concatBADGE1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatLCMD1':
            #self.generate_batch = self.mutate_randomly
            print("concatLCMD1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatrand1':
            #self.generate_batch = self.mutate_randomly
            print("concatrand1 doesnt have a generation method, will pass.")
        # elif generation_method == 'BatchBALD':
        #     self.generate_batch = self.mutate_randomly
        elif generation_method == 'concatTEMPALCMD1':
            #self.generate_batch = self.mutate_randomly
            print("concatLCMD1 doesnt have a generation method, will pass.")
        elif generation_method == 'concatTEMPArand1':
            #self.generate_batch = self.mutate_randomly
            print("concatrand1 doesnt have a generation method, will pass.")
        # elif generation_method == 'BatchBALD':
        #     self.generate_batch = self.mutate_randomly

        elif generation_method == 'Costmixrand1':
            #self.generate_batch = self.mutate_randomly
            print("Costmixrand1 doesnt have a generation method, will pass.")            
        elif generation_method == 'CostmixLCMD1':
            #self.generate_batch = self.mutate_randomly
            print("Costmixrand1 doesnt have a generation method, will pass.")     
        elif generation_method == 'CostmixTEMPArand1':
            #self.generate_batch = self.mutate_randomly
            print("Costmixrand1 doesnt have a generation method, will pass.")            
        elif generation_method == 'CostmixTEMPALCMD1':
            #self.generate_batch = self.mutate_randomly
            print("Costmixrand1 doesnt have a generation method, will pass.")     
        elif generation_method == 'PriceHundredLCMD':
            #self.generate_batch = self.mutate_randomly
            print("Costmixrand1 doesnt have a generation method, will pass.")   
        elif generation_method == 'Price20KLCMD':
            #self.generate_batch = self.mutate_randomly
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

        elif generation_method == 'salfirstlayer':
            self.generate_batch =self.mutate_by_saliencyfirstlayer
        
        elif generation_method == 'hessian':
            self.generate_batch =self.mutate_by_hessian
        elif generation_method == 'hessian_y':
            self.generate_batch =self.mutate_by_hessian
        
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
        elif generation_method == 'potts': #AC
            self.generate_batch = self.potts_chain
        
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

        elif generation_method == 'motifembed': #AC
            self.generate_batch = self.motifembed

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
            
    def potts_chain(self, x_source, n_to_make, cycles=1): #AC
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        for c in range(cycles):
            #to_mutate = self.make_mutants(to_mutate,mutation_number)
            to_mutate=originals #TEMP
        return(torch.Tensor(to_mutate))
    
    #def motifembed(self,n_to_make,seq_length,file_path,core_names):
    #    #proposed_X=gen_synth_embed.make_sim_seqs(num_seq = 5000,seq_length = 230, file_path='./pfm_AC_all.txt',core_names=['HNF4G'])
    #    proposed_X=gen_synth_embed.make_sim_seqs(num_seq = n_to_make,seq_length = seq_length, file_path=file_path,core_names=core_names)
    #    return proposed_X
    def motifembed(self,X_source,file_path,core_names):
        proposed_X=torch.empty(0)
        for i,x in enumerate(X_source):            
            new_x=gen_synth_embed.make_sim_seqs(x, num_seq = 1,seq_length = x.shape[1], file_path=file_path, #make_JASPAR_txt.py
                             #core_names=['GATA2'], 
                             #core_names=['KLF5','KLF15','NFYA','NFYC','FOXI1','FOXJ2',
                             #            'GATA2','GATA3'
                             #            ],
                             core_names=core_names,
                             #motif_freqs=motif_freqs,
                             verbose=False)
            proposed_X=torch.cat((proposed_X,new_x),axis=0)
        return proposed_X


    def vanilla_diffusion(self, model_path, 
                          n_to_make,
                          #h5file, 
                          #oracle_ckpt,
                          #chosen_model,
                          ranker,
                          seqlength=249,
                          steps=128,
                          #batch_size=128
                          
                          ):
        #python run_sample.py --model_path MODEL_PATH --steps STEPS
        
        #parser = argparse.ArgumentParser(description="Generate some samples")
        #parser.add_argument("--model_path", default="", type=str) #Need to provide model path
        #parser.add_argument("--batch_size", type=int, default=256)
        #parser.add_argument("--steps", type=int, default=249) #Sequence length
        #args = parser.parse_args()       
        device = torch.device('cuda')
        #diff_model, graph, noise = load_model_local(model_path, device)
        diff_model, graph, noise = load_model.load_model_local(model_path, device)

        #Deepstarr
        #filepath = os.path.join('../../Occasio_Dev/inputs/LentiMPRA_processed_for_dal.h5') #AC
        """
        filepath=os.path.join(h5file) #VANILLA/CONDITIONING
        data = h5py.File(filepath, 'r') #VANILLA/CONDITIONING
        """
        #ckpt_aug_path = os.path.join('../Occasio_Dev/inputs/oracle_ResidualBind_LentiMPRA_processed_for_dal_relustandard1.ckpt') 
        #ckpt_aug_path = os.path.join(oracle_ckpt)       

        #We select test data to calculate MSE and generate samples. Change if required
        """
        X_test = torch.tensor(np.array(data['X_test'])) #VANILLA/CONDITIONING
        y_test = torch.tensor(np.array(data['Y_test'])) #VANILLA/CONDITIONING
        X_test = torch.argmax(X_test, dim=1) #VANILLA/CONDITIONING
        testing_ds = TensorDataset(X_test, y_test) #VANILLA/CONDITIONING
        test_ds = torch.utils.data.DataLoader(testing_ds, batch_size=batch_size, shuffle=False, num_workers=4) #VANILLA/CONDITIONING #Anirban: num_worker>1: still same gpu
        #deepstarr = PL_ResidualBind.load_from_checkpoint(ckpt_aug_path, input_h5_file=filepath).eval() #AC
        #deepstarr=eval("PL_"+chosen_model+"(input_h5_file='"+h5file+"', initial_ds='+str(initial_ds)+', extra_str='"+extra_str+"')") # QUIQUIURG doesnt change anything in terms of pred? it only counts the ckpt you use?? #QUIQUINOURG not args
        """
        
        val_pred_seq = []
        #for _, (batch, val_target) in enumerate(test_ds): #VANILLA/CONDITIONING
        #for _ in range(n_to_make):
        for bbatch in np.array_split(range(n_to_make), ceil(len(range(n_to_make))/ranker.batch_size)):
            #sampling_fn = sampling.get_pc_sampler(graph, noise, (batch.shape[0], 249), 'analytic', steps, device=device)
            sampling_fn = sampling.get_pc_sampler(graph, noise, (1, seqlength), 'analytic', steps, device=device)
            sample = sampling_fn(diff_model) #VANILLA/CONDITIONING
            #sample = sampling_fn(diff_model, val_target.to(device)) #VANILLA/CONDITIONING
            seq_pred_one_hot = torch.nn.functional.one_hot(sample, num_classes=4).float()
            #print(f"{seq_pred_one_hot.shape=}")
            val_pred_seq.append(seq_pred_one_hot)
        val_pred_seqs = torch.cat(val_pred_seq, dim=0)
        print(f"VANILLADIFF DEBUG: {val_pred_seqs.shape=}")
        val_pred_seqs=val_pred_seqs.permute(0,2,1)
        
        #val_score = deepstarr.predict_custom(deepstarr.X_test.to(device))
        #val_pred_score = deepstarr.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))
        #sp_mse = (val_score - val_pred_score) ** 2
        #mean_sp_mse = torch.mean(sp_mse).cpu()
        #print(f"all-sp-mse {mean_sp_mse}")
        #np.savez(os.path.join(args.model_path, f"sample_{rank}.npz", ), val_pred_seqs.cpu())
        return val_pred_seqs

    def diffusion_file(self, model_path, 
                          n_to_make,
                          h5file, 
                          #oracle_ckpt,
                          #chosen_model,
                          ranker,
                          seqlength=249,
                          steps=128,
                          #batch_size=128
                          ): #DONE
        device = torch.device('cuda')
        diff_model, graph, noise = load_model.load_model_local(model_path, device)

        filepath=os.path.join(h5file) #V/CONDITIONING
        data = h5py.File(filepath, 'r') #V/CONDITIONING
        y_test = torch.tensor(np.array(data['Y_test'])) #V/CONDITIONING
        
        if n_to_make>len(y_test):
            ntimes=int(n_to_make/len(y_test))
            n__to_make=len(y_test)
        else:
            ntimes=1
            n__to_make=n_to_make

        val_pred_seq = []
        for itime in range(ntimes):
            for bbatch in np.array_split(range(n__to_make), ceil(len(range(n__to_make))/ranker.batch_size)):
                sampling_fn = sampling.get_pc_sampler(graph, noise, (1, seqlength), 'analytic', steps, device=device)
                #sample = sampling_fn(diff_model) #VANILLA/C
                #sample = sampling_fn(diff_model, val_target.to(device)) #V/CONDITIONING
                sample = sampling_fn(diff_model, y_test[bbatch].to(device)) #V/CONDITIONING
                seq_pred_one_hot = torch.nn.functional.one_hot(sample, num_classes=4).float()
                val_pred_seq.append(seq_pred_one_hot)
        val_pred_seqs = torch.cat(val_pred_seq, dim=0)
        print(f"VANILLADIFF DEBUG: {val_pred_seqs.shape=}")
        val_pred_seqs=val_pred_seqs.permute(0,2,1)
        
        return val_pred_seqs

    def diffusion_y(self, model_path, 
                          n_to_make,
                          #h5file, 
                          #oracle_ckpt,
                          #chosen_model,
                          ranker,
                          seqlength=249,
                          steps=128,
                          #batch_size=128
                          ycond=2.0,
                          ): #DONE
        device = torch.device('cuda')
        diff_model, graph, noise = load_model.load_model_local(model_path, device)
        
        val_pred_seq = []
        #for _, (batch, val_target) in enumerate(test_ds): #V/CONDITIONING
        for bbatch in np.array_split(range(n_to_make), ceil(len(range(n_to_make))/ranker.batch_size)):
            sampling_fn = sampling.get_pc_sampler(graph, noise, (1, seqlength), 'analytic', steps, device=device)
            #sample = sampling_fn(diff_model) #VANILLA/C
            #sample = sampling_fn(diff_model, val_target.to(device)) #V/CONDITIONING
            sample = sampling_fn(diff_model, torch.tensor(ycond).unsqueeze(0).unsqueeze(0).to(device)) #V/CONDITIONING
            seq_pred_one_hot = torch.nn.functional.one_hot(sample, num_classes=4).float()
            val_pred_seq.append(seq_pred_one_hot)
        val_pred_seqs = torch.cat(val_pred_seq, dim=0)
        print(f"DIFF_Y DEBUG: {val_pred_seqs.shape=}")
        val_pred_seqs=val_pred_seqs.permute(0,2,1)
        
        return val_pred_seqs


    def evoaug(self, x_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=0.05): #cycles=1): #AC
        # max_augs_per_seq : Maximum number of augmentations to apply to each sequence. Value is superceded by the number of augmentations in augment_list.
        # self.max_augs_per_seq = np.minimum(max_augs_per_seq, len(augment_list)) # default: max_augs_per_seq=2 https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf#scrollTo=WOS4yxXwWrxN
        # self.max_num_aug = len(augment_list)

        # https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        originals=torch.tensor(originals)
        print(f"{originals.shape=}")
        want_pad=False #QUIQUIURG  grep "which will be applied to pad other sequences with random DNA." so should always be FALSE???
        #if 'mut' in self.generation_method:
        if self.generation_method=='evoaugmut':
            augment_list = [
                augment.RandomMutation_up_to_ratio_Jack(mutate_frac=mutate_frac),
            ]
            maxaugperseq=1
        else: 
            augment_list = [
                ###augment.RandomDeletion(delete_min=0, delete_max=10), #25% of 39 approx to 40
                ###augment.RandomRC(rc_prob=0.5),
                ####augment.RandomInsertion(insert_min=0, insert_max=10),
                ###augment.RandomTranslocation(shift_min=0, shift_max=5),
                ####augment.RandomNoise(noise_mean=0, noise_std=0.3),
                ###augment.RandomMutation(mutate_frac=mutate_frac),
                augment.RandomDeletion(delete_min=0, delete_max=20),
                augment.RandomRC(rc_prob=0.5),
                #augment.RandomInsertion(insert_min=0, insert_max=20),
                augment.RandomTranslocation(shift_min=0, shift_max=20),
                #augment.RandomNoise(noise_mean=0, noise_std=0.2),
                #augment.RandomMutation(mutate_frac=0.05),
                #augment.RandomMutation(mutate_frac=mutate_frac),
                augment.RandomMutation_up_to_ratio_Jack(mutate_frac=mutate_frac),
            ]
            maxaugperseq=max_augs_per_seq
        max_num_aug=len(augment_list) # line 45 of: https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py 
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=hard_aug, max_augs_per_seq=max_augs_per_seq, max_num_aug=max_num_aug)
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=1, max_num_aug=1) #WORKS #QUIQUIURG is this too limited? #GOODOLD
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=max_augs_per_seq, max_num_aug=1) #DOESNT WORK
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=1, max_num_aug=max_num_aug) #DOESNT WORK
        x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=maxaugperseq, max_num_aug=max_num_aug) #DOESNT WORK
        if x_new.shape!=originals.shape: 
            print(f"ERROR: EvoAug augmentations produce different lengths: {x_new.shape=} {originals.shape=}")
            exit()
        #return(torch.Tensor(x_new))
        return x_new

    def original_evoaug(self, x_source, n_to_make, hard_aug=True, max_augs_per_seq=2, mutate_frac=0.05): #cycles=1): #AC
        # max_augs_per_seq : Maximum number of augmentations to apply to each sequence. Value is superceded by the number of augmentations in augment_list.
        # self.max_augs_per_seq = np.minimum(max_augs_per_seq, len(augment_list)) # default: max_augs_per_seq=2 https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf#scrollTo=WOS4yxXwWrxN
        # self.max_num_aug = len(augment_list)

        # https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        originals=torch.tensor(originals)
        print(f"{originals.shape=}")
        want_pad=False #QUIQUIURG  grep "which will be applied to pad other sequences with random DNA." so should always be FALSE???
        #if 'mut' in self.generation_method:
        augment_list = [
                ###augment.RandomDeletion(delete_min=0, delete_max=10), #25% of 39 approx to 40
                ###augment.RandomRC(rc_prob=0.5),
                ####augment.RandomInsertion(insert_min=0, insert_max=10),
                ###augment.RandomTranslocation(shift_min=0, shift_max=5),
                ####augment.RandomNoise(noise_mean=0, noise_std=0.3),
                ###augment.RandomMutation(mutate_frac=mutate_frac),
                augment.RandomDeletion(delete_min=0, delete_max=20),
                augment.RandomRC(rc_prob=0.5),
                #augment.RandomInsertion(insert_min=0, insert_max=20),
                augment.RandomTranslocation(shift_min=0, shift_max=20),
                #augment.RandomNoise(noise_mean=0, noise_std=0.2),
                #augment.RandomMutation(mutate_frac=0.05),
                augment.RandomMutation(mutate_frac=mutate_frac),
                #augment.RandomMutation_up_to_ratio_Jack(mutate_frac=mutate_frac),
            ]
        maxaugperseq=max_augs_per_seq
        max_num_aug=len(augment_list) # line 45 of: https://github.com/p-koo/evoaug/blob/master/evoaug/evoaug.py 
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=hard_aug, max_augs_per_seq=max_augs_per_seq, max_num_aug=max_num_aug)
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=1, max_num_aug=1) #WORKS #QUIQUIURG is this too limited? #GOODOLD
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=max_augs_per_seq, max_num_aug=1) #DOESNT WORK
        #x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=1, max_num_aug=max_num_aug) #DOESNT WORK
        x_new=evoaug_augmentations._apply_augment(originals, augment_list, want_pad=want_pad, hard_aug=True, max_augs_per_seq=maxaugperseq, max_num_aug=max_num_aug) #DOESNT WORK
        if x_new.shape!=originals.shape: 
            print(f"ERROR: EvoAug augmentations produce different lengths: {x_new.shape=} {originals.shape=}")
            exit()
        #return(torch.Tensor(x_new))
        return x_new


    def dinuc(self, x_source, n_to_make, cycles=1, batch_size=100): #AC
        # ../Lowhang/src/FUNCTIONS_FOR_SYNTHGEN_AC.py
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        originals=torch.tensor(originals)
        #to_mutate = np.array(originals)
        ##print(f"AACC: {len(to_mutate)=} {seqs=} {ranker.batch_size=}")
        X_new=torch.empty(0)
        #seqs, nts, pos =  to_mutate.shape
        #for batch in np.split(to_mutate, ceil(seqs/batch_size)):
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
            print(f"CONTROL OVER SELECT_FROM_SOURCE: {x_source.shape=} {n_to_make=}")
            idx = self.rng.choice(x_source.shape[0], n_to_make, replace=replace)
            selected=x_source[np.sort(idx)]
        elif method=='pristine':
            selected=x_source[:n_to_make] #QUIQUIURG I think this should be fine
        elif method=='highest':
            preds=ranker.Predictive_Models[0].predict_custom(x_source) #QUIQUIURG should not just be [0]
            idx=(np.argsort(preds)[::-1])[:n_to_make]
            selected=x_source[np.sort(idx)]
        return selected
    
    """
    def select_from_file(self, chosen_model, file_to_open, n_to_make, x_source, method='random', load_from_file=False, ALcycle=0, path_to_alr_inds='./fromfile_alrinds.npy'): #, replace = False, ranker=None):
        x_source_t=torch.tensor(x_source)
        data = h5py.File(file_to_open, 'r')
        if chosen_model=='ResidualBind':
            X_imported=torch.tensor(np.array(data['X_train'])) ##.requires_grad_(True)
            #y_imported=torch.tensor(np.array(data['Y_train'])) ##.requires_grad_(True)                               #ResidualBind
            #self.X_test=torch.tensor(np.array(data['X_test'])) ##.requires_grad_(True)
            #self.y_test=torch.tensor(np.array(data['Y_test'])) ##.requires_grad_(True)
            #self.X_valid=torch.tensor(np.array(data['X_valid'])) ##.requires_grad_(True)
            #self.y_valid=torch.tensor(np.array(data['Y_valid'])) ##.requires_grad_(True)
            #self.X_test2=self.X_test ##.requires_grad_(True)
            #self.y_test2=self.y_test ##.requires_grad_(True)
        elif chosen_model=='DeepSTARR':
            X_imported = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
            #self.X_train = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
            #self.y_train = torch.tensor(np.array(data['Y_train']))[:,0].unsqueeze(1)
        else:
            print("ERROR in select_from_file: wrong chosen model")
            exit()
        found=0
        if not load_from_file:
            idxs=list(np.arange(len(X_imported)))
        else:
            #if os.path.isfile(path_to_alr_inds):
            if ALcycle!=0:
                idxs=list(np.load(path_to_alr_inds))
            else:
                #os.system('rm '+path_to_alr_inds) #QUIQUIURG is this necessary? or not anymore, since I introduced the ifs with AL=0 everywhere?
                idxs=list(np.arange(len(X_imported)))
        #print(len(idxs))
        #exit()
        if ALcycle==0 or not load_from_file:
            new_X=torch.empty(0)
            #while found<n_to_make: #QUIQIUNONURG this method is a little unelegant
            while found<n_to_make and len(idxs)>0: # in this way should idxs be fully explored before found==n_to_make, you can produce stop #QUIQIUNONURG this method is a little unelegant 
                #idxs = self.rng.choice(x_source.shape[0], n_to_make, replace=replace)
                #idx = self.rng.choice(x_source.shape[0], n_to_make, replace=replace)
                #selected=x_source[np.sort(idx)]
                if method=='random':
                    idx=self.rng.choice(idxs)
                xx=X_imported[idx]
                already_present=False
                for i in range(len(x_source)):
                #for i in tqdm.tqdm(range(len(x_source)),total=len(x_source)):
                    #print(f"FROMFILE: {type(xx)=}")
                    #print(f"FROMFILE: {type(x_source)=}")
                    #print(f"FROMFILE: {xx.shape=}")
                    #print(f"FROMFILE: {x_source[i].shape=}")
                    #print(f"FROMFILE: {x_source_t[i].shape=}")
                    #xx.shape=torch.Size([4, 39])
                    #x_source[i].shape=torch.Size([4, 39])
                    if (xx==x_source_t[i]).all(): 
                        #print((xx==x_source[i]).all())
                        already_present=True
                        #print(xx)
                        #print(x_source[i])
                        #exit()
                #print(already_present)
                if not already_present: 
                    found+=1
                    #print(found,"/",n_to_make)
                    #idxs.remove(idx)
                    new_X=torch.cat((new_X,xx.unsqueeze(0)),axis=0)
                idxs.remove(idx) # AC: at the end, idx must be removed where the sequence was already present or not: if it was, it should not be considered again. If it was not, the sequence will be added and thus from now on that idx would return already_present True and thus would slow things down if considered.
        
        else: #iAL!=0
            n_to_pick=n_to_make
            print(f"FROMFILE: {ALcycle=} {n_to_pick=} {len(idxs)=}")
            if n_to_pick>len(idxs): 
                n_to_pick=len(idxs)
            proposed_idxs=np.random.choice(idxs,size=n_to_pick,replace=False)
            new_X=torch.tensor(X_imported[proposed_idxs])
            for ii in proposed_idxs:
                idxs.remove(ii)
        
        if load_from_file:
            np.save(path_to_alr_inds,np.array(idxs))
        
        print(f"FROMFILE: just before returning: {len(idxs)=}")

        return new_X

        #x_source: to make sure there are no repetitions
        """
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
                print(f"A2CHEKCFROMFILE into pool_gen.py: will load: {path_to_alr_inds_AL0=}")
            else:
                idxs=list(np.load(path_to_alr_inds))
                print(f"A2CHEKCFROMFILE into pool_gen.py: will load: {path_to_alr_inds=}")
            print(f"A2CHEKCFROMFILE into pool_gen.py: just loaded, with always_from_zero=False {ALcycle=} {len(idxs)=} {len(np.unique(idxs))=}")
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
        
        # pvals = ((1- to_mutate[seqs.flatten(),:, positions.flatten()]).astype('float64')/3.0)
        pvals = ((1- to_mutate[seqs.flatten(),:, positions.flatten()]).astype('float64'))
        #print(f'SequenceProposer.make_mutants {pvals.shape=}') #(5000,4)
        ##pvals = pvals/pvals.sum(axis=1) #ValueError: operands could not be broadcast together with shapes (5000,4) (5000,)  # Handle the case of Ns by normalising by the sum of PValues by position instead of assuming 3 non wt nucleotides
        for i_nt in range(pvals.shape[0]): #QUIQUINONURG this is not optimal probably.
            pvals[i_nt,:]/=pvals.sum(axis=1)[i_nt]
            # to see that it works:
            # a=self.rng.random(4)
            # b=np.ones(4)
            # b[2]=2
            # a[:]*=b
        mutations = self.rng.multinomial(1, pvals, size=(n_to_make*mutation_numbers) )
        to_mutate[seqs.flatten(),:, positions.flatten()] = mutations
        
        #assert (np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1)==2*mutation_numbers).all(), 'Mutation numbers are incorect' #QUIQUINONURG this needs to be reinserted, in a way that Ns (0,0,0,0) do not mess with it.
        return(to_mutate)
    
    def mutate_randomly(self, n_to_make, x_source, mutation_number, cycles=1, ranker=None):
        if self.track_time:
            start_time = time()
        
        to_mutate_final=torch.empty(0) #NTOMAKELARGER
        nrounds=int(n_to_make/len(x_source)) #NTOMAKELARGER
        print(f"in mutate_randomly: {n_to_make=}")
        print(f"in mutate_randomly: {len(x_source)=}")
        print(f"in mutate_randomly: {nrounds=}")
        n2make=len(x_source)
        print(f"in mutate_randomly: {n2make=}")
        print(f"TEST {int(n_to_make/len(x_source))=}") # preliminary test for NTOMAKELARGER
        if nrounds==0: 
            nrounds=1
            n2make=n_to_make
        for iround in range(nrounds): #NTOMAKELARGER

            #originals = self.select_from_source(n_to_make, x_source, replace = False)
            originals = self.select_from_source(n2make, x_source, replace = False) #NTOMAKELARGER
            print(f"in mutate_randomly: {len(originals)=}")
            if self.track_uncertanties:            
                #des = ranker.calculate_desiderata(originals).detach().cpu().numpy()
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
                
            #NTOMAKELARGER: QUIQUIURG use a while to make sure all sequences are different
            print(f"in mutate_randomly: {to_mutate.shape=}")
            print(f"in mutate_randomly: {to_mutate_final.shape=}")
            to_mutate_final=torch.cat((to_mutate_final,torch.Tensor(to_mutate)),axis=0) #NTOMAKELARGER

        return to_mutate_final #NTOMAKELARGER
        #return(torch.Tensor(to_mutate))
    
    def saliency_one_step(self, to_optimize, mutation_numbers, ranker, temp): #, to_backprop='unc'):
        
        #print(f"AACC: {to_optimize=} {to_optimize.shape=}")
        #print(f"AACC: {mutation_numbers=} ")
        #print(f"AACC:  {ranker.batch_size=} ")
        #print(f"AACC: {ranker.uncertainty_method=} ")
        #print(f"AACC: {ranker.uncertainty_weight=}")
        #print(f"AACC: {ranker.how_many_new_batches_at_once=} ")

        # Copy array
        to_mutate = np.array(to_optimize)
        
        seqs, nts, pos =  to_mutate.shape
        # batch_uncs = []
        batch_sails = []
        [m.zero_grad() for m in ranker.Predictive_Models]
        ##print(f"AACC: {len(to_mutate)=} {seqs=} {ranker.batch_size=}")

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
                #print('x: ',x)
                x.retain_grad()
                #print('x: ',x)

                if self.generation_method=='saliency_aleat':
                    preds_av = ranker.get_pred(x, keep_grads = True)
                elif self.generation_method=='saliency_evidential':
                    preds_av, aleat_uncs, epist_uncs = ranker.pred_unc_evidential(x, keep_grads = True)
                else:
                    unc_all, preds_av = ranker.calculate_desiderata(x, keep_grads = True)
                    # if np.isnan(unc_all).sum()>0: #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    #     print(f"NANCHECK: {np.isnan(unc_all).sum()=}")
                    # if np.isinf(unc_all).sum()>0:
                    #     print(f"NANCHECK: is inf {np.isinf(unc_all).sum()=}")
                    # if (unc_all==0).any():
                    #     print(f"NANCHECK: ==0")

                #print(f"AACC unc_all: {unc_all.shape=} {unc_all=}")
                # batch_uncs.append(unc_all.detach().cpu().numpy())

                #if to_backprop=='unc':
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
            #for index in range(len(to_mutate)):
            #    #grad=PL_Models_interpr_utils.captum_gradientshap_no_y(to_mutate,model=ranker.Predictive_Models[0],index=index, todevice=False, apply_corr=True, device='cuda') #QUIQUIURG when I'll need to do it for all Predictive_Models: ranker.calculate_desiderata: preds_av=allj_preds.mean(axis=0) #average across models So if averaging and backpropagation are commutable, I could average the shapley scores. If they are not commutable its a problem bc in principle I should act within the tangermeme function!!
            #    grad=PL_Models_interpr_utils.captum_gradientshap_no_y(to_mutate,model=ranker.Predictive_Models[0],index=index, todevice=True, apply_corr=True, device='cuda') #QUIQUIURG when I'll need to do it for all Predictive_Models: ranker.calculate_desiderata: preds_av=allj_preds.mean(axis=0) #average across models So if averaging and backpropagation are commutable, I could average the shapley scores. If they are not commutable its a problem bc in principle I should act within the tangermeme function!!
            #    batch_sails.append(grad.detach().cpu().numpy())
            #for index in range(len(to_mutate)):
            #    grad=PL_Models_interpr_utils.captum_gradientshap_no_y(np.expand_dims(to_mutate[index,:,:],axis=0),model=ranker.Predictive_Models[0],index=0, todevice=False, apply_corr=True, device='cuda') #QUIQUIURG when I'll need to do it for all Predictive_Models: ranker.calculate_desiderata: preds_av=allj_preds.mean(axis=0) #average across models So if averaging and backpropagation are commutable, I could average the shapley scores. If they are not commutable its a problem bc in principle I should act within the tangermeme function!!
            #    batch_sails.append(grad)
            #grad=PL_Models_interpr_utils.captum_gradientshap_no_y_no_index(to_mutate,model=ranker.Predictive_Models[0],todevice=False, apply_corr=True, null_method='standard', device='cuda', num_background = 1000)
            #batch_sails.append(grad)
            #for bbatch in np.array_split(to_mutate, ceil(len(to_mutate)/ranker.batch_size)):
            for bbatch in np.array_split(to_mutate, ceil(len(to_mutate)/10)):
                #grad=PL_Models_interpr_utils.captum_gradientshap_no_y_no_index(bbatch,model=ranker.Predictive_Models[0],todevice=False, apply_corr=True, null_method='standard', device='cuda', num_background = 1000) #works but slow! bc of todevice=False I think
                grad=PL_Models_interpr_utils.captum_gradientshap_no_y_no_index(bbatch,model=ranker.Predictive_Models[0],todevice=True, apply_corr=True, null_method='standard', device='cuda', num_background = 1000)
                batch_sails.append(grad.detach().cpu().numpy())

        elif 'DeepLiftSHAP' in self.generation_method:
            for index in range(len(to_mutate)):
                X_attr=deep_lift_shap(ranker.Predictive_Models[0], torch.tensor(to_mutate), target=index, random_state=self.seed)
                batch_sails.append(X_attr)

        #unc_all = np.concatenate(batch_uncs)
        unc_sail = np.concatenate(batch_sails)
        #print('unc_sail (concated): ',unc_sail)
        #np.save('unc_sail_'+ranker.outflag+'.npy',unc_sail)
        #print(f"{unc_sail.shape=}")
        
        
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
            #This failes with more than one mutation. will trey psuedo flooring
#             scale_thresh = np.log(1e-40)*temp
#             mask = (unc_sail<=scale_thresh).all(axis=-1).all(axis=-1)
#             mins = unc_sail[mask].min(axis=-1).min(axis=-1)
#             unc_sail[mask] = unc_sail[mask]-(mins-scale_thresh)[:,None,None]
            
            #QUIQUI This is a hacky solution to stop overflow errors in the np.exp should probably fix this
            scale_thresh = np.log(1e+40)*temp            
            mask = (unc_sail>=scale_thresh).any(axis=-1).any(axis=-1)
            maxes = unc_sail[mask].max(axis=-1).max(axis=-1)
            unc_sail[mask] = unc_sail[mask]-(maxes-scale_thresh)[:,None,None] 
                 
            unc_sail[old_seq_mask] = -np.inf #QUIQUI setting old sequence probabilities to zero this is prob fine
            ex = np.exp(unc_sail/temp)
            ex = ex + 1e-10
            ex[old_seq_mask] = 0
#             ex = (ex*np.logical_not(to_mutate)).astype('float64')
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
        
        ##import sys
        ##np.set_printoptions(threshold=sys.maxsize)
        ##np.set_printoptions(threshold=np.inf) # https://stackoverflow.com/questions/1987694/how-do-i-print-the-full-numpy-array-without-truncation
        ##print(f"AACC: {(np.not_equal(to_optimize, to_mutate).sum(axis=1).sum(axis=1))=} {2*mutation_numbers=} {np.where((np.not_equal(to_optimize, to_mutate).sum(axis=1).sum(axis=1)!=2*mutation_numbers))=}")
        
        #test = (np.not_equal(to_optimize, to_mutate).sum(axis=1).sum(axis=1)==2*mutation_numbers) #QUIQUINONURG this needs to be reinserted, in a way that Ns (0,0,0,0) do not mess with it
        #assert test.all(), 'Mutation numbers are incorect\n%s'%str(test) #QUIQUINONURG this needs to be reinserted, in a way that Ns (0,0,0,0) do not mess with it.
        return(to_mutate)
        
    def mutate_by_saliency(self, n_to_make, x_source, cycles, mutations_per, ranker, temp, decay=None): #, to_backprop='unc'):
        if self.track_time:
            start_time = time()
        
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        if self.track_uncertanties:
            #des = ranker.calculate_desiderata(originals).detach().cpu().numpy()
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
        
        ##print(f"AC{cycles=}")
        for c in range(cycles):
            if (decay is None) or (temp == 'neg_inf'):
                t = temp
            elif cycles>1:
                t = decay + (temp-decay)*((((cycles-1)-c)/(cycles-1))**2) # calculate temperature for accepting mutations
            else:
                t = decay
            
            #if 'saliency' in self.generation_method:
            to_mutate = self.saliency_one_step(to_mutate, mutations_per, ranker, t) #, to_backprop=to_backprop)
            #elif 'GradientSHAP' in self.generation_method:
            #    to_mutate = self.saliency_one_step(to_mutate, mutations_per, ranker, t) #, to_backprop=to_backprop)

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

    #""
    def hessian_one_step(self, to_optimize, not_to_obtain, mutation_numbers, ranker, temp, to_backprop='y'):
    #def hessian_one_step(self, to_optimize, not_to_obtain, mutation_numbers, ranker, temp, to_backprop='unc'):

        # Copy array
        to_mutate = np.array(to_optimize)
        
        seqs, nts, pos =  to_mutate.shape
        batch_hessians=[]
        #[m.zero_grad() for m in ranker.Predictive_Models]
        
        #for batch in np.split(to_mutate, ceil(seqs/ranker.batch_size)):
        #    x = torch.tensor(batch).float().requires_grad_()
        #    print('x: ',x)
        #    x.retain_grad()
        
        for i in range(seqs):
        
            x = to_mutate[i]
            x = torch.tensor(x).unsqueeze(0).float().requires_grad_()

            x.retain_grad()
            
            if to_backprop=='y':
                hess=torch.autograd.functional.hessian(ranker.Predictive_Models[0].predict_custom,x[0].unsqueeze(0)) #QUIQUIURG average over all Predictive Models #QUIQUISOLVED? torch.autograd.functional.hessian can in principle have create_graph but shouldnt be necessary unless for third order derivative
            elif to_backprop=='unc':
                hess=torch.autograd.functional.hessian(ranker.calculate_desiderata_4Hess,x[0].unsqueeze(0))

            [m.zero_grad() for m in ranker.Predictive_Models]

            batch_hessians.append(hess)

            #if temp == 'neg_inf':
            for __ in ['dummy']: #QUIQUINONURG

                def multidim_argsort_hess(hess, temperature=1):
                    flat=[]
                    indexes=[]
                    boltz=[]
                    for i0 in range(hess.shape[0]):
                        for i1 in range(hess.shape[1]):
                            for i2 in range(hess.shape[2]):
                                for i3 in range(hess.shape[3]):
                                    if not(i0==i2 or i1==i3): # otherwise the pair of index couples indicates the same nt to mutate
                                        flat.append(hess[i0,i1,i2,i3].item())
                                        indexes.append([i0,i1,i2,i3])
                                        if temperature=='neg_inf':
                                            #print("NEGING!!")
                                            boltz.append(0.0) # dummy

                                        else:
                                            boltz.append(np.exp(-hess[i0,i1,i2,i3].item()/temperature))
                    boltz=np.array(boltz)
                    if temperature!='neg_inf':
                        boltz/=np.sum(boltz, axis=0)
                    order=np.argsort(flat)[::-1]
                    #print(f"{order[0:10]=}")
                    #print(f"{flat[0:10]=}")
                    #print(f"{boltz[0:10]=}")
                    #print(f"{len(order)=}")
                    #print(f"{len(flat)=}")
                    indexes_ordered=np.array(indexes)[order]
                    flat_ordered=np.array(flat)[order]
                    #flat_ordered=flat[0:10]
                    #flat_ordered=flat[order[-1]]
                    #print(f"{flat_ordered=}")
                    boltz_ordered=boltz[order]
                    #print(f"{boltz_ordered.shape=}")
                    #exit()
                    return indexes_ordered, flat_ordered, boltz_ordered #np.array(indexes)[order], flat[order], boltz[order]

                sq_hess=hess.squeeze(3).squeeze(0) # shape=(1, 4, 39, 1, 4, 39) -> (1, 4, 39, 4, 39)
                argsorts,flat_hess,boltz=multidim_argsort_hess(sq_hess, temperature=temp)

                if temp=='neg_inf':
                    i_mut=0
                else:
                    print(f"{argsorts[0]=}")
                    #i_mut=self.rng.choice(argsorts, p=boltz) #ValueError: a must be 1-dimensional
                    #chosen_index=self.rng.choice(np.arange(len(argsorts)), p=boltz)
                    #i_mut=argsorts[chosen_index]
                    i_mut=self.rng.choice(np.arange(len(argsorts)), p=boltz)
                accepted=0
                while accepted<mutation_numbers:
                    idx=argsorts[i_mut] #hessian indexes corresponding to greatest hessian (i_mut=0), second greatest (i_mut=1), ...
                    prev_ind1=np.argmax(to_mutate[i,:,idx[1]]) #argmax does return the index for the 1
                    prev_ind2=np.argmax(to_mutate[i,:,idx[3]])
                    if (not_to_obtain==None).any:
                        condition=(not (prev_ind1==idx[0] or prev_ind2==idx[2]))
                    else:
                        print(f"{not_to_obtain=} {not_to_obtain.any==None=} {not_to_obtain.all==None=}")
                        old_ind1=np.argmax(not_to_obtain[i,:,idx[1]])
                        old_ind2=np.argmax(not_to_obtain[i,:,idx[3]])
                        condition=(not (prev_ind1==idx[0] or prev_ind2==idx[2] or old_ind1==idx[0] or old_ind2==idx[2]))
                        
                    #if not (prev_ind1==idx[0] or prev_ind2==idx[2]):
                    #if not (prev_ind1==idx[0] or prev_ind2==idx[2] or old_ind1==idx[0] or old_ind2==idx[2]):
                    if condition:
                        to_mutate[i,:,idx[1]]=np.zeros(4) #to_mutate.shape: (100, 4, 39)
                        to_mutate[i,idx[0],idx[1]]=1.
                        to_mutate[i,:,idx[3]]=np.zeros(4)
                        to_mutate[i,idx[2],idx[3]]=1.
                        accepted+=1
                    #else:
                    #    #print(f"{i_mut=} {prev_ind1=} {idx[0]=} {prev_ind2=} {idx[2]=}")
                    #    #exit()
                    if temp == 'neg_inf':
                        i_mut+=1
                    else:
                        i_mut=self.rng.choice(np.arange(len(argsorts)), p=boltz) #QUIQUIURG should you remove the previously explored i_mut??? or it doesnt change anything, since it will simply be rejected again?

                for_verif=(np.not_equal(to_optimize[i], to_mutate[i])).sum(axis=1).sum()
                verif = for_verif==4*mutation_numbers
                assert verif,'Mutation numbers are incorrect: %s'%str(i)+' '+str(np.not_equal(to_optimize[i], to_mutate[i]))+' '+str(for_verif)
            #else:
            #    print("ERROR: temp!=neg_inf not implemented yet.")
            #    exit()
            [m.zero_grad() for m in ranker.Predictive_Models] #QUIQUIURG is this necessary here?
        return(to_mutate)
    #""
    def hessian_one_step_2backprop(self, to_optimize, not_to_obtain, mutation_numbers, ranker, temp, to_backprop='y'):
    #def hessian_one_step_2backprop(self, to_optimize, not_to_obtain, mutation_numbers, ranker, temp, to_backprop='unc'):

        # Copy array
        to_mutate = np.array(to_optimize)
        
        seqs, nts, pos =  to_mutate.shape
        batch_hessians=[]
        [m.zero_grad() for m in ranker.Predictive_Models]
        
        """ test with quick_proposer.py """

        for batch in np.split(to_mutate, ceil(seqs/ranker.batch_size)):
            x = torch.tensor(batch).float().requires_grad_()
            #print(f"{x.shape=}")
            x.retain_grad()

            unc_all, preds_av = ranker.calculate_desiderata(x, keep_grads = True)
            if to_backprop=='y':
                preds_av.mean().backward(create_graph=True) #QUIQUIURG preds_av should be the right thing, even after a quick look into calculate_desiderata, but would better double check
            elif to_backprop=='unc':
                unc_all.mean().backward(create_graph=True) # The mean acts over the batch as well, but since the partial derivative of a sum y = x1 + x2 + ... wrt x1 considers x2 etc as constants, this is not a problem
            
            sail = x.grad # sail.shape=torch.Size([batch_size, 4, L])

            print(f"{sail.shape=}")
            print("AC: cannot reduce sail to sail.mean or anything: would have to implement a for loop, or maybe even a for loop over a dataloader. But I am not sure if this reintroduces the problem of batches (nor if I can skip multidim_argsort which is inherently non-batchable)")

            ##hess=np.zeros((sail.shape[0],sail.shape[1]*sail.shape[2],sail.shape[1]*sail.shape[2]))
            ##for nt in range(sail.shape[1]):
            ##    for l in range(sail.shape[2]):
            ##        x.grad.data.zero_() # If we do not release the 1-order gradient before calculating the 2-oder derivative, the result will be the addition between the 1-order derivative and the 2-oder derivative. 
            ##        sail[:,nt,l].backward()
            ##        hess[:,nt,l] = x.grad.data.cpu().numpy()
            ##        print("HERE",hess.shape)
            ##        [m.zero_grad() for m in ranker.Predictive_Models]
            
            #x.grad.data.zero_()
            ##sail.view(sail.shape[0],sail.shape[1]*sail.shape[2],sail.shape[1]*sail.shape[2]).mean().backward()
            ##sail.view(sail.shape[0],sail.shape[1]*sail.shape[2]).mean().backward()
            ##hess = x.grad #.data
            ##print(f"pre: {hess.shape=}")
            ##hess=hess.view(sail.shape[0],sail.shape[1],sail.shape[2],sail.shape[1],sail.shape[2])
            ##print(f"post: {hess.shape=}")
            ##hess=hess.cpu().numpy()
            
            """ last chat with Jack: it should be batchable if the mean is taken over batches only, which can be done by flattening to batch, 4L, 4L"""
            hess=torch.zeros((sail.shape[0],sail.shape[1]*sail.shape[2],sail.shape[1]*sail.shape[2]))
            sail1=sail.view(sail.shape[0],sail.shape[1]*sail.shape[2])
            #print(f"{sail.grad=}")
            for i_s in range(sail1.shape[1]):
                x.grad.data.zero_()
                sail1[i_s].mean().backward()
                grd=x.grad.data.view(sail.shape[0],sail.shape[1]*sail.shape[2]) #.cpu().numpy()
                hess[:,i_s,:]=grd
                hess[:,:,i_s]=grd
                """ LAST: need to find a way to save first backprop but not second. 
                - Maybe by doing it within a def??? 
                - Or maybe by using some [:,] for considering the whole batch at once?"""
                exit()
            hess=hess.view(sail.shape[0],sail.shape[1],sail.shape[2],sail.shape[1],sail.shape[2]).cpu().numpy()
            [m.zero_grad() for m in ranker.Predictive_Models]
            batch_hessians.append(hess)
            exit()

        # unc_all = np.concatenate(batch_uncs)
        hessian = np.concatenate(batch_hessians)
        if np.sum(hessian)==0.0:
            print("ERROR: hessian resulting in only-zeros.")
            exit()

        #print('unc_sail (concated): ',unc_sail)

        # HHHH
        exit()
        if temp == 'neg_inf':
            
            # make WT saliencies minimal to stop selection
            cut = np.array(hessian)
            cut[to_mutate.astype(bool)] = cut.min()-1
            
            # identify the maximal nucleotide and position combos
            max_nts = cut.argmax(axis=1, keepdims=False)
            i, j = np.indices((cut.shape[0],cut.shape[2]))
            parted_poses = cut[i,max_nts, j].argpartition(-mutation_numbers,axis=1,)
            
            #mutate the array
            to_mutate[i[:,-mutation_numbers:], :, parted_poses[:,-mutation_numbers:]] = 0
            to_mutate[i[:,-mutation_numbers:], max_nts[i,parted_poses][:,-mutation_numbers:], parted_poses[:,-mutation_numbers:]] = 1

        """
        if temp == 'neg_inf':
            
            # make WT saliencies minimal to stop selection
            cut = np.array(unc_sail)
            cut[to_mutate.astype(bool)] = cut.min()-1
            
            # identify the maximal nucleotide and position combos
            max_nts = cut.argmax(axis=1, keepdims=False)
            i, j = np.indices((cut.shape[0],cut.shape[2]))
            parted_poses = cut[i,max_nts, j].argpartition(-mutation_numbers,axis=1,)
            
            #mutate the array
            to_mutate[i[:,-mutation_numbers:], :, parted_poses[:,-mutation_numbers:]] = 0
            to_mutate[i[:,-mutation_numbers:], max_nts[i,parted_poses][:,-mutation_numbers:], parted_poses[:,-mutation_numbers:]] = 1
        """

        """
        for i in range(seqs):
        
            x = to_mutate[i]
            x = torch.tensor(x).unsqueeze(0).float().requires_grad_()

            x.retain_grad()
            
            if to_backprop=='y':
                hess=torch.autograd.functional.hessian(ranker.Predictive_Models[0].predict_custom,x[0].unsqueeze(0)) #QUIQUIURG average over all Predictive Models #QUIQUISOLVED? torch.autograd.functional.hessian can in principle have create_graph but shouldnt be necessary unless for third order derivative
            elif to_backprop=='unc':
                hess=torch.autograd.functional.hessian(ranker.calculate_desiderata_4Hess,x[0].unsqueeze(0))

            [m.zero_grad() for m in ranker.Predictive_Models]

            batch_hessians.append(hess)

            #if temp == 'neg_inf':
            for __ in ['dummy']: #QUIQUINONURG

                def multidim_argsort_hess(hess, temperature=1):
                    flat=[]
                    indexes=[]
                    boltz=[]
                    for i0 in range(hess.shape[0]):
                        for i1 in range(hess.shape[1]):
                            for i2 in range(hess.shape[2]):
                                for i3 in range(hess.shape[3]):
                                    if not(i0==i2 or i1==i3): # otherwise the pair of index couples indicates the same nt to mutate
                                        flat.append(hess[i0,i1,i2,i3].item())
                                        indexes.append([i0,i1,i2,i3])
                                        if temperature=='neg_inf':
                                            #print("NEGING!!")
                                            boltz.append(0.0) # dummy

                                        else:
                                            boltz.append(np.exp(-hess[i0,i1,i2,i3].item()/temperature))
                    boltz=np.array(boltz)
                    if temperature!='neg_inf':
                        boltz/=np.sum(boltz, axis=0)
                    order=np.argsort(flat)[::-1]
                    #print(f"{order[0:10]=}")
                    #print(f"{flat[0:10]=}")
                    #print(f"{boltz[0:10]=}")
                    #print(f"{len(order)=}")
                    #print(f"{len(flat)=}")
                    indexes_ordered=np.array(indexes)[order]
                    flat_ordered=np.array(flat)[order]
                    #flat_ordered=flat[0:10]
                    #flat_ordered=flat[order[-1]]
                    #print(f"{flat_ordered=}")
                    boltz_ordered=boltz[order]
                    #print(f"{boltz_ordered.shape=}")
                    #exit()
                    return indexes_ordered, flat_ordered, boltz_ordered #np.array(indexes)[order], flat[order], boltz[order]

                sq_hess=hess.squeeze(3).squeeze(0) # shape=(1, 4, 39, 1, 4, 39) -> (1, 4, 39, 4, 39)
                argsorts,flat_hess,boltz=multidim_argsort_hess(sq_hess, temperature=temp)

                if temp=='neg_inf':
                    i_mut=0
                else:
                    print(f"{argsorts[0]=}")
                    #i_mut=self.rng.choice(argsorts, p=boltz) #ValueError: a must be 1-dimensional
                    #chosen_index=self.rng.choice(np.arange(len(argsorts)), p=boltz)
                    #i_mut=argsorts[chosen_index]
                    i_mut=self.rng.choice(np.arange(len(argsorts)), p=boltz)
                accepted=0
                while accepted<mutation_numbers:
                    idx=argsorts[i_mut] #hessian indexes corresponding to greatest hessian (i_mut=0), second greatest (i_mut=1), ...
                    prev_ind1=np.argmax(to_mutate[i,:,idx[1]]) #argmax does return the index for the 1
                    prev_ind2=np.argmax(to_mutate[i,:,idx[3]])
                    if (not_to_obtain==None).any:
                        condition=(not (prev_ind1==idx[0] or prev_ind2==idx[2]))
                    else:
                        print(f"{not_to_obtain=} {not_to_obtain.any==None=} {not_to_obtain.all==None=}")
                        old_ind1=np.argmax(not_to_obtain[i,:,idx[1]])
                        old_ind2=np.argmax(not_to_obtain[i,:,idx[3]])
                        condition=(not (prev_ind1==idx[0] or prev_ind2==idx[2] or old_ind1==idx[0] or old_ind2==idx[2]))
                        
                    #if not (prev_ind1==idx[0] or prev_ind2==idx[2]):
                    #if not (prev_ind1==idx[0] or prev_ind2==idx[2] or old_ind1==idx[0] or old_ind2==idx[2]):
                    if condition:
                        to_mutate[i,:,idx[1]]=np.zeros(4) #to_mutate.shape: (100, 4, 39)
                        to_mutate[i,idx[0],idx[1]]=1.
                        to_mutate[i,:,idx[3]]=np.zeros(4)
                        to_mutate[i,idx[2],idx[3]]=1.
                        accepted+=1
                    #else:
                    #    #print(f"{i_mut=} {prev_ind1=} {idx[0]=} {prev_ind2=} {idx[2]=}")
                    #    #exit()
                    if temp == 'neg_inf':
                        i_mut+=1
                    else:
                        i_mut=self.rng.choice(np.arange(len(argsorts)), p=boltz) #QUIQUIURG should you remove the previously explored i_mut??? or it doesnt change anything, since it will simply be rejected again?

                for_verif=(np.not_equal(to_optimize[i], to_mutate[i])).sum(axis=1).sum()
                verif = for_verif==4*mutation_numbers
                assert verif,'Mutation numbers are incorrect: %s'%str(i)+' '+str(np.not_equal(to_optimize[i], to_mutate[i]))+' '+str(for_verif)
            #else:
            #    print("ERROR: temp!=neg_inf not implemented yet.")
            #    exit()
            [m.zero_grad() for m in ranker.Predictive_Models] #QUIQUIURG is this necessary here?
        return(to_mutate)
        """
        
    def mutate_by_hessian(self, n_to_make, x_source, x_previous,
                          cycles, mutations_per, ranker, temp, decay=None): #, to_backprop='y'):
        """
        if self.track_time:
            start_time = time()
        """
            
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        if x_previous!=None:
            previous = self.select_from_source(n_to_make, x_previous, replace = False)
        else:
            previous = None
        """
        if self.track_uncertanties:
            des = ranker.calculate_desiderata(originals).detach().cpu().numpy()

            tmp = self.tracker.get('ave_unc', [])
            tmp.append(des.mean())
            self.tracker['ave_unc'] = tmp

            tmp = self.tracker.get('std_unc', [])
            tmp.append(des.std())
            self.tracker['std_unc'] = tmp
        """
            
        """
        if self.track_batches:

            tmp = self.tracker.get('batch', [])
            tmp.append(originals)
            self.tracker['batch'] = tmp
        """
        
        """
        if self.track_hamming:
            tmp = self.tracker.get('ham', [])
            tmp.append(self.get_hamming(originals))
            self.tracker['ham'] = tmp
        """
            
        to_mutate = np.array(originals)
        #to_mutate = np.array(originals.detach().cpu().numpy())
        if previous!=None: 
            not_to_obtain = np.array(previous)
        else:
            #not_to_obtain=np.array([None]*len(to_mutate))
            not_to_obtain=np.array([None]) 
            #not_to_obtain=np.array([None]*len(to_mutate[0][0]))

        for c in range(cycles):
            if (decay is None) or (temp == 'neg_inf'):
                t = temp
            elif cycles>1:
                t = decay + (temp-decay)*((((cycles-1)-c)/(cycles-1))**2) # calculate temperature for accepting mutations
            else:
                t = decay

            if self.generation_method=='hessian':
                to_backprop='unc'
            elif self.generation_method=='hessian_y':
                to_backprop='y'

            #to_mutate = self.hessian_one_step(to_mutate, mutations_per, ranker, t)
            to_mutate = self.hessian_one_step(to_mutate, not_to_obtain, mutations_per, ranker, t, to_backprop=to_backprop)
            #to_mutate = self.hessian_one_step_2backprop(to_mutate, not_to_obtain, mutations_per, ranker, t, to_backprop=to_backprop)
            
            """
            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp
            """

            """
            if self.track_uncertanties:
                des = ranker.calculate_desiderata(to_mutate).detach().cpu().numpy()

                tmp = self.tracker.get('ave_unc', [])
                tmp.append(des.mean())
                self.tracker['ave_unc'] = tmp

                tmp = self.tracker.get('std_unc', [])
                tmp.append(des.std())
                self.tracker['std_unc'] = tmp
            """

            """
            if self.track_batches:
                
                tmp = self.tracker.get('batch', [])
                tmp.append(to_mutate)
                self.tracker['batch'] = tmp
            """

            """
            if self.track_hamming:
                tmp = self.tracker.get('ham', [])
                tmp.append(self.get_hamming(to_mutate))
                self.tracker['ham'] = tmp
            """
                
            """
            if self.track_time or self.track_uncertanties or self.track_batches or self.track_hamming:
                with open(self.track_pref+'.pckl', 'wb') as f:
                    pickle.dump(self.tracker, f)
            """
                
        return(torch.Tensor(to_mutate))      
     
    def saliencyfirstlayer_one_step(self, to_optimize, mutation_numbers, ranker, temp):
        
        # Copy array
        to_mutate = np.array(to_optimize)
        
        seqs, nts, pos =  to_mutate.shape
        # batch_uncs = []
        batch_sails = []
        [m.zero_grad() for m in ranker.Predictive_Models]
        for batch in np.split(to_mutate, ceil(seqs/ranker.batch_size)):
            x = torch.tensor(batch).float().requires_grad_()
#             print('x: ',x)
            x.retain_grad()
#             print('x: ',x)

            unc_all, preds_av = ranker.calculate_desiderata(x, keep_grads = True)
            #print('unc_all: ',unc_all)
            # batch_uncs.append(unc_all.detach().cpu().numpy())
            unc_all.mean().backward()

            if ranker.uncertainty_method!='sigma_deep_ensemble':
                unc_sail = ranker.Predictive_Models[0].model.first_layer_grad.detach().cpu().numpy() # AC: must be a self of the Model
            else:
                print("ERROR: not implemented yet")
                exit()


            #unc_sail = x.grad.data.cpu().numpy()
#             print('unc_sail: ',unc_sail)
            batch_sails.append(unc_sail)
            [m.zero_grad() for m in ranker.Predictive_Models]

        # unc_all = np.concatenate(batch_uncs)
        unc_sail = np.concatenate(batch_sails)
#         print('unc_sail (concated): ',unc_sail)
        
        
        """
        if temp == 'neg_inf':
            
            # make WT saliencies minimal to stop selection
            cut = np.array(unc_sail)
            cut[to_mutate.astype(bool)] = cut.min()-1
            
            # identify the maximal nucleotide and position combos
            max_nts = cut.argmax(axis=1, keepdims=False)
            i, j = np.indices((cut.shape[0],cut.shape[2]))
            parted_poses = cut[i,max_nts, j].argpartition(-mutation_numbers,axis=1,)
            
            #mutate the array
            to_mutate[i[:,-mutation_numbers:], :, parted_poses[:,-mutation_numbers:]] = 0
            to_mutate[i[:,-mutation_numbers:], max_nts[i,parted_poses][:,-mutation_numbers:], parted_poses[:,-mutation_numbers:]] = 1
                        
        elif isinstance(temp, float) or isinstance(temp, int):
            unc_sail = unc_sail.astype('float64')
            old_seq_mask = to_mutate.astype(bool)
            unc_sail[old_seq_mask] = 0 #QUIQUI stoping old sequences from messing up the scaling should probably fix this
            
            #QUIQUI This is a hacky solution to stop low numbers from all becoming zero in the np.exp should probably fix this
            #This failes with more than one mutation. will trey psuedo flooring
#             scale_thresh = np.log(1e-40)*temp
#             mask = (unc_sail<=scale_thresh).all(axis=-1).all(axis=-1)
#             mins = unc_sail[mask].min(axis=-1).min(axis=-1)
#             unc_sail[mask] = unc_sail[mask]-(mins-scale_thresh)[:,None,None]
            
            #QUIQUI This is a hacky solution to stop overflow errors in the np.exp should probably fix this
            scale_thresh = np.log(1e+40)*temp            
            mask = (unc_sail>=scale_thresh).any(axis=-1).any(axis=-1)
            maxes = unc_sail[mask].max(axis=-1).max(axis=-1)
            unc_sail[mask] = unc_sail[mask]-(maxes-scale_thresh)[:,None,None] 
                 
            unc_sail[old_seq_mask] = -np.inf #QUIQUI setting old sequence probabilities to zero this is prob fine
            ex = np.exp(unc_sail/temp)
            ex = ex + 1e-10
            ex[old_seq_mask] = 0
#             ex = (ex*np.logical_not(to_mutate)).astype('float64')
            probs = (ex/ex.sum(axis=1).sum(axis=1)[:,None,None])
            badmask = np.isnan(probs).any(axis=-1).any(axis=-1)

            pos_to_mut = np.stack([self.rng.choice(np.arange(probs.shape[2]), p=probs[i].sum(axis=0), size = mutation_numbers,replace=False) for i in range(probs.shape[0])])
            i, j = np.indices(pos_to_mut.shape)

            pos_to_mut = pos_to_mut.flatten()
            i = i.flatten()

            to_mutate[i, :, pos_to_mut] = self.rng.multinomial(1, probs[i, :, pos_to_mut]/probs[i, :, pos_to_mut].sum(axis=1)[:,None])

        else:
            assert False, "invalid temp"
        """
#         print((np.not_equal(to_optimize, to_mutate).sum(axis=1).sum(axis=1)))
        test = (np.not_equal(to_optimize, to_mutate).sum(axis=1).sum(axis=1)==2*mutation_numbers)
        #assert test.all(), 'Mutation numbers are incorect\n%s'%str(test)
        exit()
        return(to_mutate)
        
    def mutate_by_saliencyfirstlayer(self, n_to_make, x_source, cycles, mutations_per, ranker, temp, decay=None ):
        """
        if self.track_time:
            start_time = time()
        """

        originals = self.select_from_source(n_to_make, x_source, replace = False)
        """
        if self.track_uncertanties:
            des = ranker.calculate_desiderata(originals).detach().cpu().numpy()

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
        """

        to_mutate = np.array(originals)
        
        for c in range(cycles):
            if (decay is None) or (temp == 'neg_inf'):
                t = temp
            elif cycles>1:
                t = decay + (temp-decay)*((((cycles-1)-c)/(cycles-1))**2) # calculate temperature for accepting mutations
            else:
                t = decay
                
            to_mutate = self.saliencyfirstlayer_one_step(to_mutate, mutations_per, ranker, t)

            """
            if self.track_time:
                end_time = time()
                tmp = self.tracker.get('time', [])
                tmp.append(end_time-start_time)
                self.tracker['time'] = tmp

            if self.track_uncertanties:
                des = ranker.calculate_desiderata(to_mutate).detach().cpu().numpy()

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
            """

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
        #uncs = ranker.calculate_desiderata(to_mutate).detach().cpu().numpy()
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
                #unc_new = ranker.calculate_desiderata(x).detach().cpu().numpy()
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
                
#             if not (np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1)==2*mutations_per_cycle*(cycle+1)).all():
#                 print('Mutation numbers are incorect')
#                 print(np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1))
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
            #des = ranker.calculate_desiderata(originals).detach().cpu().numpy()
            #des = ranker.calculate_desiderata(originals)[0].detach().cpu().numpy()
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
        # WEAREHERE
        new_array=np.ones((seqs, nts, pos))*-99999.
        #cut[to_mutate.astype(bool)] = cut.min()-1 
        
        for mut in range(mutation_number):
            print('working on mut: ',mut)
            batch_uncs = []
            ##aaa=0
            ##bbb=0
            ##ccc=self.gen_all_singles_batches(to_mutate, ranker.batch_size, self.mutable_window)
            ##print(f"{ccc.shape=}")
            for batch in self.gen_all_singles_batches(to_mutate, ranker.batch_size, 
                                                      self.mutable_window):
                x = torch.tensor(batch).float()
                #print(f"{x.shape=} {mut}/{mutation_number}")
                ##if x.shape[0]==100: aaa+=1
                ##if x.shape[0]==99: bbb+=1
                #batch_uncs.append(ranker.calculate_desiderata(x).detach().cpu().numpy())
                if self.generation_method=='greedy_y':
                    batch_uncs.append(ranker.calculate_desiderata(x)[1].detach().cpu().numpy())
                else:
                    batch_uncs.append(ranker.calculate_desiderata(x)[0].detach().cpu().numpy())
            ##print(f"{aaa=} {bbb=} {aaa*100+bbb*99}")
            #print('batches done')
            unc_all = np.concatenate(batch_uncs)
            print(f"greedy: {unc_all.shape=}")
            new_array[np.where(np.logical_not(to_mutate))]=unc_all
            #print('batches done')
            
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
            
        #if not (np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1)==2*mutation_number).all():
        #    print('Mutation numbers are incorect')
        #    print('np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1): ',np.not_equal(originals, to_mutate).sum(axis=1).sum(axis=1))
        return(torch.Tensor(to_mutate))
    
    def genetic(self, n_to_make, x_source, mutation_number, cycles, x_prob, expansion_fold, ranker):
        if self.track_time:
            start_time = time()
            
        originals = self.select_from_source(n_to_make, x_source, replace = False)
        if self.track_uncertanties:
            #des = ranker.calculate_desiderata(originals).detach().cpu().numpy()
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
#             print('new_mutants.shape: ',new_mutants.shape)
#             print('len(new_mutants): ',len(new_mutants))
#             print('batch_size: ',batch_size)
#             print('ceil(len(new_mutants)/batch_size): ',ceil(len(new_mutants)/batch_size))
#             print('len(new_mutants)/ceil(len(new_mutants)/batch_size): ',len(new_mutants)/ceil(len(new_mutants)/batch_size))

            #unc_all = ranker.calculate_desiderata(new_mutants).detach().cpu().numpy()
            #unc_all = ranker.calculate_desiderata(new_mutants)[0].detach().cpu().numpy() #goodold pre 5 Jan 2024
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
        return(to_mutate)
        
    def batch_from_file(self):
        assert False, "Not implemented"