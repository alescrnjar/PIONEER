import os
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
#import pytorch_lightning as pl #DSRR
#import tfomics
#from Bio.motifs.matrix import PositionSpecificScoringMatrix
#from Bio.Seq import Seq
import tqdm
import random

#torch.manual_seed(41) #These are only applied when this script is first imported, credo QUIQUI
#random.seed(41)
#np.random.seed(41)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
def one_hot_to_tokens(one_hot):
    #""
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    #""
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def one_hot_to_tokens_TT(one_hot):
    #""
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    #""
    #tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    #tokens=torch.empty(0).to(device)
    #for i in range(one_hot.shape[0]):
    #    tokens=torch.cat((tokens,one_hot.shape[1]))
    #print(one_hot.shape)
    tokens = torch.tile(torch.tensor(one_hot.shape[1]), (one_hot.shape[0],)) #.to(device)  # Vector of all D #https://pytorch.org/docs/stable/generated/torch.tile.html
    seq_inds, dim_inds = torch.where(one_hot) #.to(device)
    tokens[seq_inds] = dim_inds
    return tokens
"""

def one_hot_to_1to4_tokens_TT(one_hot):
    #tokens=torch.zeros(one_hot.shape[1],dtype=torch.int)
    tokens=torch.zeros(one_hot.shape[1],dtype=torch.float)
    seq_inds, dim_inds = torch.where(one_hot) #.to(device)
    #seq_inds=torch.tensor(seq_inds,dtype=torch.int)
    seq_inds=torch.tensor(seq_inds,dtype=torch.float)
    tokens[dim_inds] = seq_inds
    return tokens

def ohe_to_seq(x,four_zeros_ok=False):
    seq=''
    for i in range(x.shape[1]):
        """
        #if int(x[0][i])==1:
        if int(x[0][i])==1 and int(x[1][i])==0 and int(x[2][i])==0 and int(x[3][i])==0:
            seq+='A'
        #elif int(x[1][i])==1:
        elif int(x[0][i])==0 and int(x[1][i])==1 and int(x[2][i])==0 and int(x[3][i])==0:
            seq+='C'
        #elif int(x[2][i])==1:
        elif int(x[0][i])==0 and int(x[1][i])==0 and int(x[2][i])==1 and int(x[3][i])==0:
            seq+='G' #'T' #QUIQUI
        #elif int(x[3][i])==1:
        elif int(x[0][i])==0 and int(x[1][i])==0 and int(x[2][i])==0 and int(x[3][i])==1:
            seq+='T' #'G'
        """
        if int(x[0,i])==1 and int(x[1,i])==0 and int(x[2,i])==0 and int(x[3,i])==0:
            seq+='A'
        elif int(x[0,i])==0 and int(x[1,i])==1 and int(x[2,i])==0 and int(x[3,i])==0:
            seq+='C'
        elif int(x[0,i])==0 and int(x[1,i])==0 and int(x[2,i])==1 and int(x[3,i])==0:
            seq+='G' #'T' #QUIQUI
        elif int(x[0,i])==0 and int(x[1,i])==0 and int(x[2,i])==0 and int(x[3,i])==1:
            seq+='T' #'G'
        else:
            if int(x[0,i])==0 and int(x[1,i])==0 and int(x[2,i])==0 and int(x[3,i])==0:
                if four_zeros_ok:
                    seq+='N'
                else:
                    ##print("- ohe_to_seq: Error:",i,float(x[0][i]),str(x[0][i]),str(x[1][i]),str(x[2][i]),str(x[3][i]))
                    #print("- ohe_to_seq: Error:",i,float(x[0][i]),float(x[1][i]),float(x[2][i]),float(x[3][i])
                    print("- ohe_to_seq: Error at point",i,":",float(x[0,i]),float(x[1,i]),float(x[2,i]),float(x[3,i]))
                    exit()
            else:
                print("- ohe_to_seq: Error at point",i,":",float(x[0,i]),float(x[1,i]),float(x[2,i]),float(x[3,i]))
                exit()
    return seq

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)
def dna_to_one_hot(dna):
    return np.identity(4)[
        np.unique(string_to_char_array(dna), return_inverse=True)[1]
    ]


def plot_distrib(data,flag='_',nbins=100, outdir='./'):
    """ AC: Plot histogram of a distribution. """
    fig = plt.figure(1, figsize=(4, 4))
    plt.hist(data, bins=nbins, density=True)
    outfile=outdir+'Hist_'+flag+'.png'
    fig.savefig(outfile,dpi=150)
    #plt.show()
    #print("DONE: "+outfile)
    plt.clf()