from scipy import stats
import numpy as np
import torch
import matplotlib.pyplot as plt

def custom_pcc(x,y):
    """normal pcc, implemented by me"""
    xav=np.mean(x)
    yav=np.mean(y)
    num=0.0
    denx=0.0
    deny=0.0
    for i in range(len(x)):
        num+=(x[i]-xav)*(y[i]-yav)
        denx+=(x[i]-xav)**2
        deny+=(y[i]-yav)**2
    den=np.sqrt(denx*deny)
    pcc=num/den
    return pcc

"""
def wpearsonr(x,y):
    xav=np.mean(x)
    yav=np.mean(y)
    num=0.0
    denx=0.0
    deny=0.0
    for i in range(len(x)):
        num+=x[i]*(x[i]-xav)*(y[i]-yav)
        denx+=x[i]*(x[i]-xav)**2
        deny+=x[i]*(y[i]-yav)**2
    den=np.sqrt(denx*deny)
    pcc=num/den
    return pcc
"""
def wpearsonr(x,y,m=0.,M=1.): # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    if type(x)==torch.Tensor:
        x=x.detach().cpu().numpy()
        x=np.array(x)
    if type(y)==torch.Tensor:
        y=y.detach().cpu().numpy()
        y=np.array(y)
    #print("------------------------")
    #print(x,type(x))
    #print(np.min(x))
    #transl=abs(np.min(x))
    #weights=x+transl # translation required to avoid negative weights which cannot be accepted (of which amount it does not matter, I verified it, but it may be useful if it were the same for all datasets, and min isnt)
    weights=(x-m)/(M-m)
    #print(weights)
    if min(weights)<0: 
        print("ERROR: min and/or max of range for WPCC miscalibrated: minimum weight is:",min(weights))
        exit()
    #print(f"{x=}")
    #print(f"{weights=} {np.min(x)=}")
    #weights=np.arange(len(x))
    cov=np.cov(x,y,aweights=weights)
    num=cov[0][1]
    """
    cov(a,a)  cov(a,b)
    cov(a,b)  cov(b,b)
    """
    den=np.sqrt(cov[0][0]*cov[1][1])
    #print(f"{num=}")
    pcc=num/den
    return pcc

def control_wpearsonr(x,y): # https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation
    transl=abs(np.min(x)) 
    #transl=9.
    #transl=99.
    weights=x+transl # translation required to avoid negative weights which cannot be accepte. (of which amount it does not matter, I verified it, but it may be useful if it were the same for all datasets, and min isnt)
    #print(f"{weights=}")
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
    #print(f"{np.sum(weights)=}")
    #print(f"---------------{weights.shape=}")
    weights/=np.sum(weights)
    #print(f"{((x-y)**2).shape=}")
    wmse=np.average((x-y)**2,weights=weights)
    return wmse

def control_weighted_mse(x,y,m=0.,M=1.):
    weights=(x-m)/(M-m)
    ##weights=np.ones(len(x))
    wmse=0.
    for i in range(len(x)):
        wmse+=weights[i]*(x[i]-y[i])**2/np.sum(weights)
    return wmse

if __name__=='__main__':
    np.random.seed(41)

    ytrue=[]
    ypred=[]
    first=0

    size=100
    #factor=100
    factor=10
    noise1=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    noise2=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    ytrue.append(np.arange(first,first+size)+noise1)
    ypred.append(np.arange(first,first+size)+noise2)
    first+=size

    size=100
    factor=50
    noise1=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    noise2=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    ytrue.append(np.arange(first,first+size)+noise1)
    ypred.append(np.arange(first,first+size)+noise2)
    first+=size

    size=100
    #factor=10
    factor=100
    noise1=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    noise2=-1.*factor*np.random.random(size)+factor*np.random.random(size)
    ytrue.append(np.arange(first,first+size)+noise1)
    ypred.append(np.arange(first,first+size)+noise2)
    first+=size

    ytrue=np.concatenate(ytrue)
    ypred=np.concatenate(ypred)

    #print(f"{ytrue=}")  
    pcc=stats.pearsonr(ytrue,ypred)[0]
    my_pcc=custom_pcc(ytrue,ypred)
    wpcc=wpearsonr(ytrue,ypred,m=min(ytrue),M=max(ytrue))
    wpcc_control=control_wpearsonr(ytrue,ypred)
    print(f"{pcc=} {my_pcc=} {wpcc=} {wpcc_control=}")

    wmse=weighted_mse(ytrue,ypred,m=np.min(ytrue),M=np.max(ytrue))
    wmse_control=control_weighted_mse(ytrue,ypred,m=np.min(ytrue),M=np.max(ytrue))
    print(f"{wmse=} {wmse_control=}")

    plt.scatter(ytrue,ypred)
    plt.show()

    exit()

    chosen_dataset='VTS1_rnacompete2009labels_q085_5000'
    chosen_model='ResidualBind'
    input_h5_file='../inputs/'+chosen_dataset+'.h5'
    import h5py
    from PL_Models import *
    data = h5py.File(input_h5_file, 'r')
    ckptfile='../inputs/oracle_ResidualBind_rnacompete2009_processed_for_dal.ckpt'
    model=eval('PL_'+chosen_model+'(input_h5_file="'+input_h5_file+'",initial_ds=True)')
    model = model.load_from_checkpoint(ckptfile, input_h5_file=input_h5_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred=model.predict_custom(model.X_test[0:10].to(device))
    wpcc=model.metrics(model.y_test[0:10],pred)['WPCC']
    print(f"Model: {wpcc=}")


    """ DONE: use weighted mean https://numpy.org/doc/stable/reference/generated/numpy.average.html
               and weighted cov: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
               
               see formula for rho(X Y) here: https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation
               """