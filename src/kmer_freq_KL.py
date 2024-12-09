import os
import numpy as np
import tqdm
from FUNCTIONS_4_DALdna import ohe_to_seq, dna_to_one_hot

def kmer_frequencies_AC(X,min_k=1,max_k=3,verbose=False): #min_k=3: dinucleotide frequencies should be already maintained by construction, so 2 will be skipped
    import itertools 
    products={}
    if verbose: print("--- Make products")
    for k in range(min_k,max_k+1):
        if verbose: os.system('date')
        if verbose: print(k,'/',max_k) 
        list1=itertools.product('ACGT', repeat=k)
        products[k]=[]
        for tpl in list(list1):
            prod=''.join(tpl)
            products[k].append(prod)
    if verbose: os.system('date')

    if verbose: print("--- Make frequencies")
    freqs={}
    for k in range(min_k,max_k+1): 
        if verbose: os.system('date')
        if verbose: print(k,'/',max_k)
        #for i,x in tqdm.tqdm(enumerate(X), total=len(X)):
        for i,x in enumerate(X):
            #print(i,x)
            #print(i,x.shape)
            x_dna=ohe_to_seq(x) 
            for prod in products[k]:
                freqs[prod]=x_dna.count(prod) # WARNING: 'AAA'.count('AA') returns 1, not 2
    if verbose: os.system('date')

    return freqs

"""
# https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
# https://medium.com/geekculture/techniques-to-measure-probability-distribution-similarity-9145678d68a6
def kl(p, q):
    #""Kullback-Leibler divergence D(P || Q) for discrete distributions
    #Parameters
    #----------
    #p, q : array-like, dtype=float, shape=n
    #Discrete probability distributions.
    #""
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where((p != 0), p * np.log(p / q), 0))
"""

def KL_divergence(p, q, epsilon = 1e-10):
    kl = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    return kl

def KL_divergences(X,reference_distrib,min_k,max_k):
    KLdivs=np.empty(len(X))
    for i,x in enumerate(X):
        x0=np.expand_dims(x,axis=0)
        #x0=x.unsqueeze(axis=0)
        KLdivs[i]=KL_divergence(np.array(list(reference_distrib.values())),
                                np.array(list(kmer_frequencies_AC(x0,min_k=min_k,max_k=max_k).values())))
    return KLdivs
      

if __name__=='__main__':
    x_dna='ATTTGCCGGATCGA'
    x=dna_to_one_hot(x_dna).transpose(1,0)
    x0=np.expand_dims(x,axis=0)
    #print(f"{x=}")
    
    import torch
    x_s_0=torch.tensor(np.array([
        dna_to_one_hot('ATTTGCCGGATCGA').transpose(1,0), 
        dna_to_one_hot('ATTTGCCGGATCGT').transpose(1,0),
        dna_to_one_hot('CTTTGCCGGATCGT').transpose(1,0),  
        dna_to_one_hot('ATTAGCGGGAACGA').transpose(1,0),
        ]))
    x_s_1=torch.tensor(np.array([
        dna_to_one_hot('ATTTGAAGGATCGA').transpose(1,0), 
        dna_to_one_hot('ATTTGCAGGATCGT').transpose(1,0), 
        dna_to_one_hot('ATTAACGGGAACGA').transpose(1,0),
        dna_to_one_hot('ATTAACGGGAACGT').transpose(1,0),
        dna_to_one_hot('GTTAACGGGAACGA').transpose(1,0),
        ]))

    min_k=2
    max_k=4
    ref_distr=kmer_frequencies_AC(x_s_1,min_k=min_k,max_k=max_k)
    KLd=KL_divergences(x_s_0,ref_distr,min_k=min_k,max_k=max_k)
    print(KLd)

    
    """
    for x_dna in ['ATTTGCCGGATCGA','ATTTGCCGGATCGT','ATTAGCGGGAACGA','AAAAGCCAAAACGA','AAAAAAAAAAAAAA','ATTGCCGGATCGA','GGGGGGGGGGGGGG',]:
        x=dna_to_one_hot(x_dna).transpose(1,0)
        x1=np.expand_dims(x,axis=0)

        freq0=kmer_frequencies_AC(x0,min_k=2,max_k=4)
        freq1=kmer_frequencies_AC(x1,min_k=2,max_k=4)
        
        frq0=(np.array(list(freq0.values())))
        frq1=(np.array(list(freq1.values())))

        #kl=kl(frq0,frq1)
        kl=KL_divergence(frq0,frq1)
        print(kl)
    """