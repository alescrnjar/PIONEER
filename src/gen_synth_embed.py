# https://github.com/p-koo/exponential_activations/blob/master/code/generate_data/task1_generate_synthetic_dataset.ipynb

import os, sys, h5py
import numpy as np
import pandas as pd
import torch

def get_jaspar_motifs(file_path):
    def get_motif(f):
        line = f.readline()
        name = line.strip().split()[1]
        pfm = []
        for i in range(4):
            line = f.readline()
            #print(f"{line=}")
            #if line!='\n': #AC
            if len(line.split()[1]) > 1:
                pfm.append(np.asarray(np.hstack([line.split()[1][1:], line.split()[2:-1]]), dtype=float))
            else:
                pfm.append(np.asarray(line.split()[2:-1], dtype=float))
        pfm = np.vstack(pfm)
        sum_pfm = np.sum(pfm, axis=0)
        pwm = pfm/np.outer(np.ones(4), sum_pfm)
        line = f.readline()
        return name, pwm

    num_lines = sum(1 for line in open(file_path))
    num_motifs = int(num_lines/6)

    f = open(file_path)
    tf_names = []
    tf_motifs = []
    for i in range(num_motifs):
        name, pwm = get_motif(f)
        tf_names.append(name)
        tf_motifs.append(pwm)

    return tf_motifs, tf_names

def orig_generate_model(core_motifs, seq_length):                #Orig
    num_motif = len(core_motifs)
    cum_dist = np.cumsum([0, 0.5, 0.25, 0.17, 0.05, 0.3])
    # sample core motifs for each grammar                #Orig
    valid_sim = False
    while not valid_sim:
        # determine number of core motifs in a given grammar model                #Orig
        num_interactions = np.where(np.random.rand() > cum_dist)[0][-1]+1 #np.random.randint(min_interactions, max_interactions)
        # randomly sample motifs
        sim_motifs = np.random.randint(num_motif, size=num_interactions)
        num_sim_motifs = len(sim_motifs)                #Orig
        #sim_motifs = sim_motifs[np.random.permutation(num_sim_motifs)]
        # verify that distances aresmaller than sequence length
        distance = 0
        for i in range(num_sim_motifs):
            distance += core_motifs[sim_motifs[i]].shape[1]                #Orig
        if seq_length > distance > 0:
            valid_sim = True    
    # simulate distances between motifs + start                 #Orig
    valid_dist = False
    while not valid_dist:
        remainder = seq_length - distance
        sep = np.random.uniform(0, 1, size=num_sim_motifs+1)                 #Orig
        sep = np.round(sep/sum(sep)*remainder).astype(int) 
        if np.sum(sep) == remainder:
            valid_dist = True
    # build a PWM for each regulatory grammar                #Orig
    pwm = np.ones((4,sep[0]))/4
    for i in range(num_sim_motifs):
        print(f"{pwm.shape=} {core_motifs[sim_motifs[i]].shape=} {(np.ones((4,sep[i+1]))/4).shape=}")
        pwm = np.hstack([pwm, core_motifs[sim_motifs[i]], np.ones((4,sep[i+1]))/4])
    #print(f"{num_sim_motifs=} {sep=} {pwm.shape=} {seq_length=}") #commented: 3 sept 2024
    return pwm, sim_motifs


def generate_model_AC_1(core_motifs, seq_length,
                        core_names,
                        orig_ohe,
                        #motif_freqs,
                        verbose=False
                        ):
    num_motif = len(core_motifs)
    interaction_probabilities=[0, 0.5, 0.25, 0.17, 0.05, 0.3] # != len(core_names = ['Arid3a', 'CEBPB', 'FOSL1', 'Gabpa', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'SP1', 'SRF', 'STAT1', 'YY1'])
    #if verbose: print(f"{motif_freqs}=")
    #motif_probabilities=[motif_freqs[key] for key in core_names]
    #motif_probabilities=[0.0]+motif_probabilities
    cum_dist = np.cumsum(interaction_probabilities) # AC I dont think this would actually work: what if I want 2, but the last 2? it's ok bc it's just for the probability of the number of interactions!
    if verbose: print(f"{cum_dist=}")
    # sample core motifs for each grammar
    valid_sim = False
    while not valid_sim:
        # determine number of core motifs in a given grammar model
        rn=np.random.rand()
        if verbose:
            print(f"{rn=}")
            print(f"{np.where(rn > cum_dist)=}")
            print(f"{np.where(rn > cum_dist)[0]=}")
            print(f"{np.where(rn > cum_dist)[0][-1]=}")
        num_interactions = np.where(rn > cum_dist)[0][-1]+1 #np.random.randint(min_interactions, max_interactions)
        sim_motifs = np.random.randint(num_motif, size=num_interactions) # randomly sample motifs
        num_sim_motifs = len(sim_motifs) # AC by construction, shouldnt it be len(sim_motifs)==num_interactions??
        # verify that distances aresmaller than sequence length
        distance = 0
        for i in range(num_sim_motifs):
            distance += core_motifs[sim_motifs[i]].shape[1]
        if seq_length > distance > 0:
            valid_sim = True    
    # simulate distances between motifs + start 
    valid_dist = False
    while not valid_dist:
        remainder = seq_length - distance
        sep = np.random.uniform(0, 1, size=num_sim_motifs+1) #for num_sim_motifs=2: array([0.66531851, 0.37359712, 0.7810503 ])
        sep = np.round(sep/sum(sep)*remainder).astype(int) #for num_sim_motifs=2: array([1, 1, 2])
        if np.sum(sep) == remainder:
            valid_dist = True
    if verbose: print(f"{sep=}")
    # build a PWM for each regulatory grammar
    #""
    #pwm = np.ones((4,sep[0]))/4  # initialize irst bit before first motif (credo)
    pwm=orig_ohe[:,:sep[0]]
    last_index=sep[0]
    last_indices=[last_index]
    #print() #commented: 3 sept 2024
    for i in range(num_sim_motifs):
        #pwm = np.hstack([pwm, core_motifs[sim_motifs[i]], np.ones((4,sep[i+1]))/4])
        #print(f"{pwm.shape=} {last_index=} {(core_motifs[sim_motifs[i]]).shape[1]=} {sep[i+1]=}")
        #print(f"{last_index+len(core_motifs[sim_motifs[i]])=} {last_index+len(core_motifs[sim_motifs[i]])+sep[i+1]=} {len(orig_ohe)=}")
        pwm = np.hstack([pwm, core_motifs[sim_motifs[i]], orig_ohe[:,last_index+(core_motifs[sim_motifs[i]]).shape[1]:last_index+(core_motifs[sim_motifs[i]]).shape[1]+sep[i+1]]])
        last_index+=(core_motifs[sim_motifs[i]]).shape[1]+sep[i+1]
        #print(f"{last_index=}") #commented: 3 sept 2024
        last_indices.append(last_index)
    #print(f"{last_indices=}")
    #print(f"{num_sim_motifs=} {sep=} {pwm.shape=}") #commented: 3 sept 2024
    #""
    #print() #commented: 3 sept 2024
    return pwm, sim_motifs


def simulate_sequence(sequence_pwm): # AC: turn a PWM into a seq
    """simulate a sequence given a sequence model"""
    nucleotide = 'ACGT'
    seq_length = sequence_pwm.shape[1]
    Z = np.random.uniform(0,1,seq_length) # generate uniform random number for each nucleotide in sequence
    cum_prob = sequence_pwm.cumsum(axis=0) # calculate cumulative sum of the probabilities
    # go through sequence and find bin where random number falls in cumulative probabilities for each nucleotide
    one_hot_seq = np.zeros((4, seq_length))
    for i in range(seq_length):
        index=[j for j in range(4) if Z[i] < cum_prob[j,i]][0]
        one_hot_seq[index,i] = 1
    return one_hot_seq

def make_sim_seqs(orig_ohe,
                  num_seq = 5000,seq_length = 230, file_path='./pfm_AC_all.txt',
                  core_names=['GATA2'], 
                  #motif_freqs={'KLF5':0.2, 'KLF15':0.2, 'NFYA':0.2, 'NFYC':0.2, 'FOXI1':0.2, 'FOXJ2':0.2,'GATA2':0.4,'GATA3':0.4}, # fimo_on_original_ds.py
                  verbose=False):

    # parse JASPAR motifs
    #savepath = './' #./../data'
    #file_path = os.path.join(savepath, 'pfm_vertebrates.txt')
    #file_path = os.path.join(savepath, 'pfm_AC_all.txt')
    motif_set, motif_names = get_jaspar_motifs(file_path)
    ##jaspf='JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt'
    ##motif_set, motif_names = get_jaspar_motifs(jaspf)

    # get a subset of core motifs 
    #core_names = ['Arid3a', 'CEBPB', 'FOSL1', 'Gabpa', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'SP1', 'SRF', 'STAT1', 'YY1']
    """
    FIMO scans for LentiMPRA motifs: KLF5, KLF15, NFYA, NFYC, FOXI1, FOXJ2.
    HepG2: HNF4A, HNF4G. 
    K562: GATA2, GATA3.
    """
    ##core_names=['HNF4G']

    strand_motifs = []
    core_index = []
    for name in core_names:
        if name in motif_names: #AC, to solve for GATA3
            strand_motifs.append(motif_set[motif_names.index(name)])
            core_index.append(motif_names.index(name))

    # generate reverse compliments
    core_motifs = []
    for pwm in strand_motifs:
        core_motifs.append(pwm)
        reverse = pwm[:,::-1]
        core_motifs.append(reverse[::-1,:]) 
    #print(f"{core_motifs=}")

    # dataset parameters
    #num_seq = 5000             # number of sequences
    #seq_length = 230            # lsaength of sequence
    ##min_interactions = 1        # exponential rate of number of motifs for each grammar
    ##max_interactions = 5
    max_labels = len(core_names)

    # generate sythetic sequences as a one-hot representation
    seq_pwm = []
    seq_model = []    
    targets = []
    for j in range(num_seq):
        #signal_pwm, labels = orig_generate_model(core_motifs, seq_length)
        #""
        signal_pwm, labels = generate_model_AC_1(core_motifs, seq_length, 
                                                 core_names,
                                                 orig_ohe,
                                                 #motif_freqs,
                                                 verbose=verbose)
        #""
        simseq=simulate_sequence(signal_pwm)
        #print(f"{simseq=} {simseq.shape=}")
        #exit()
        seq_pwm.append(simseq)

    proposed_X=torch.tensor(np.array(seq_pwm),dtype=torch.float32)
    return proposed_X

if __name__=='__main__':
    def random_ohe_seq(seq_len):
        indices = torch.randint(0, 4, size=(1, 1, seq_len))  # Assuming 4 classes and 7 elements
        one_hot = torch.zeros(1, 4, seq_len)  # Shape: (1, 4, 7)
        one_hot.scatter_(1, indices, 1)
        return one_hot
    
    #motif_freqs={'KLF5':0.8, 'KLF15':0.2, 'NFYA':0.2, 'NFYC':0.2, 'FOXI1':0.2, #'FOXJ2':0.2,
    #             'GATA2':0.4,'GATA3':0.4} # fimo_on_original_ds.py
    import json

    # Opening JSON file
    with open('fimoorig_K562.json') as json_file:
        motif_freqs = json.load(json_file)
    print(f"{motif_freqs=}")
    #exit()

    import FUNCTIONS_4_DALdna
    np.random.seed(33) # for reproducibility
    seq_len=17 #230
    nseq=1 #5000
    orig_ohe=np.array(random_ohe_seq(seq_len).squeeze(0).numpy())
    #orig_ohe=np.transpose(np.array(random_ohe_seq(17).squeeze(0).numpy()),(1,0))
    print(f"{orig_ohe=} {orig_ohe.shape=}")
    print(FUNCTIONS_4_DALdna.ohe_to_seq(orig_ohe))
    #proposed_X=make_sim_seqs(orig_ohe, num_seq = 5000,seq_length = 230, file_path='./pfm_AC_all.txt',core_names=['HNF4G'], verbose=True) #make_JASPAR_txt.py
    proposed_X=make_sim_seqs(orig_ohe, num_seq = nseq,seq_length = seq_len, file_path='./pfm_AC_all.txt', #make_JASPAR_txt.py
                             #core_names=['GATA2'], 
                             core_names=['KLF5','KLF15','NFYA','NFYC','FOXI1','FOXJ2',
                                         'GATA2','GATA3'
                                         ],
                             #motif_freqs=motif_freqs,
                             verbose=True)
    print(f"{proposed_X.shape=}")

    print("ISSUE!!!! THERE IS NO GATA3 IN JASPAR.txt")
    print("ISSUE!!!! VALUES ARE NOT NORMALIZED, ARE COUNTS!")
    print("BUT: none of the two issue matters: I am not considering the probabilities yet, only WHAT motifs. The probabilities are just about HOW MANY")
    print("SCRIPT END")