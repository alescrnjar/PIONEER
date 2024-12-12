import torch
import math

def model_positionwise_mean(x):
    return(torch.mean(x, dim=0))

def binary_entropy(p_s, device, small=1e-10):
    #omp_s=torch.ones(p_s.shape).to(device).add(torch.multiply(p_s,-1))
    omp_s=torch.ones(p_s.shape).add(torch.multiply(p_s,-1))
    #
    log_p_s=torch.log2(p_s.add(small)) #QUIQUI should this be with a +/- depending on p_s >< 0.5?
    p_log_p_s = torch.multiply(p_s,log_p_s)
    #
    log_omp_s=torch.log2(omp_s.add(small)) #QUIQUI should this be with a +/- depending on p_s >< 0.5?
    omp_log_omp_s = torch.multiply(omp_s,log_omp_s)
    #
    entr=torch.add(p_log_p_s,omp_log_omp_s).multiply(-1)
    return entr
