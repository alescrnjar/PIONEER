from scipy import stats
import numpy as np
import torch
import matplotlib.pyplot as plt

def wpearsonr(x,y,m=0.,M=1.): # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    if type(x)==torch.Tensor:
        x=x.detach().cpu().numpy()
        x=np.array(x)
    if type(y)==torch.Tensor:
        y=y.detach().cpu().numpy()
        y=np.array(y)
    weights=(x-m)/(M-m)
    if min(weights)<0: 
        print("ERROR: min and/or max of range for WPCC miscalibrated: minimum weight is:",min(weights))
        exit()
    cov=np.cov(x,y,aweights=weights)
    num=cov[0][1]
    den=np.sqrt(cov[0][0]*cov[1][1])
    pcc=num/den
    return pcc

def control_wpearsonr(x,y): # https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation
    transl=abs(np.min(x)) 
    weights=x+transl # translation required to avoid negative weights which cannot be accepte. (of which amount it does not matter, I verified it, but it may be useful if it were the same for all datasets, and min isnt)
    mX=np.average(x,weights=weights)
    mY=np.average(y,weights=weights)
    sX=np.average((x-mX)**2,weights=weights)
    sY=np.average((y-mY)**2,weights=weights)
    sXY=np.average((x-mX)*(y-mY),weights=weights)
    return sXY/np.sqrt(sX*sY)

def weighted_mse(x,y,m=0.,M=1.,no_weights=False):
    if type(x)==torch.Tensor:
        x=x.detach().cpu().numpy()
        x=np.array(x)
    if type(y)==torch.Tensor:
        y=y.detach().cpu().numpy()
        y=np.array(y)
    if not no_weights:
        weights=(x-m)/(M-m)
    else:
        weights=np.ones(len(x))
    weights/=np.sum(weights)
    wmse=np.average((x-y)**2,weights=weights)
    return wmse

def control_weighted_mse(x,y,m=0.,M=1.):
    weights=(x-m)/(M-m)
    wmse=0.
    for i in range(len(x)):
        wmse+=weights[i]*(x[i]-y[i])**2/np.sum(weights)
    return wmse

