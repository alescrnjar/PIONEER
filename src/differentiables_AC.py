import torch
import math

def model_positionwise_mean(x):
    return(torch.mean(x, dim=0))

def AC_binary_entropy(p_s, device, small=1e-10):
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

#""
# example usage: AC_mutual_information(preds_mc, device)
def AC_mutual_information(preds, device, small=1e-10):
    mean_pred=torch.mean(preds,axis=0)
    entr_of_mean=AC_binary_entropy(mean_pred, device, small)
    entrs=torch.zeros(preds.shape)
    #print(f"{preds[0].shape=} {mean_pred.shape=}")
    for i in range(preds.shape[0]):
        entrs[i]=AC_binary_entropy(preds[i], device, small)
    mean_entr=torch.mean(entrs,axis=0)
    mutual_information=torch.subtract(entr_of_mean,mean_entr)
    #print(f"{preds=} {mean_pred=}")
    #print(f"{entr_of_mean=} {entrs=} {mean_entr=} {mutual_information=}")
    #print(f"{entr_of_mean.shape=} {entrs.shape=} {mean_entr.shape=} {mutual_information.shape=}")
    return mutual_information
#""

"""
def AC_mutual_info(p_s,device):
    entr=AC_binary_entropy(p_s,device)
    conditional_entr=torch.mean(entr) #Conditional entropy of positions over weights (approximated) #QUIQUIURG how is the conditioning to the weights implemented?
    mutual_info=torch.add(entr,conditional_entr.multiply(-1)) 
    return mutual_info
"""
    
"""
def mutual_info(x):
    # Mutual info

    # Positionwise entropy
    # average probability
    mean_prob = model_positionwise_mean(x)

    # log(p)
    log_mean_prob = torch.log2(mean_prob)

    # calculate entropy
    pmeanlogpmean = torch.multiply(mean_prob,log_mean_prob)
    predictive_entropy = torch.multiply(torch.sum(pmeanlogpmean, dim=-2), -1)

    #Conditional entropy of positions over weights (approximated)
    log_p = torch.log2(x)
    plogp = torch.multiply(x, log_p)
    position_entropy = torch.sum(torch.multiply(plogp, -1), dim=-2)
    conditional_entropy = torch.mean(position_entropy, dim=0)

    mutual_info = torch.subtract(predictive_entropy,conditional_entropy)
    return(torch.mean(mutual_info, dim=-1))
"""

if __name__=='__main__':
    import numpy as np
    batch_size=7

    device='cuda'
    torch.manual_seed(41)
    ##p_s=torch.rand((batch_size,1)).requires_grad_().to(device)
    p_s=torch.rand((batch_size)).requires_grad_() #.to(device)
    #print(f"{p_s=}")

    entr=AC_binary_entropy(p_s, device)
    #print(f"{entr=}")
    #print(f"{mutual_info=}")

    #for x in np.linspace(0.0,1.0,10):
    for y in np.linspace(0.0,1e-9,10):
        print(y,AC_binary_entropy(torch.tensor(y),device))
              
    print("\n\n-------- Mutual Info")
    #preds_mc=torch.rand((5,batch_size))

    for preds_mc_list in [
        [0.5745, 0.5804, 0.5636, 0.5590, 0.5693], # One point, with low uncertainty, centered around 0.5
        [0.1745, 0.3804, 0.6636, 0.7590, 0.9993], # One point, with high unc.
        [0.8745, 0.8804, 0.8636, 0.8590, 0.8693], # One point, with low uncertainty, centered around 0.8
    ]:
        #preds_mc=(torch.tensor([0.5]*5)+torch.rand(5).divide(10)).unsqueeze(1)
        preds_mc=torch.tensor(preds_mc_list).unsqueeze(1)
        #mut_info=mutual_info(preds_mc)
        mut_info=AC_mutual_information(preds_mc, device)
        print(f"{preds_mc=} {preds_mc.shape=}")
        print(f"{mut_info=}")