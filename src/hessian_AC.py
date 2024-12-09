import torch
import numpy as np 

"""
def calc_hessian_AC(y,x):
    sals=torch.autograd.grad(y,x, create_graph=True, retain_graph=True)[0]
    # derivative with respect to each component
    print(f"{sals=}")
    hesss=torch.zeros(x.shape[0],x.shape[0])
    for i,sal in enumerate(sals):
        hess=torch.autograd.grad(sal,x, retain_graph=True)[0]
        hesss[i]=hess
    print(f"{hesss=}")
    return hesss
"""

def calc_hessian_AC(y, x):
    print("----- into calc_hessian")
    print(f"{y=}")
    print(f"{x=}")
    sals=torch.autograd.grad(y,x, create_graph=True) #[0]
    #sals=torch.autograd.grad(y,x, create_graph=True, retain_graph=True)[0]
    #sals=torch.autograd.grad(y,x, create_graph=True, retain_graph=True, is_grads_batched=True)[0]
    #sals=torch.autograd.grad(y,x, create_graph=True, retain_graph=True, materialize_grads=True)[0]
    #sals=torch.autograd.backward(y,x, create_graph=True, retain_graph=True)[0]
    #sals=torch.autograd.backward(y,x)
    #sals=torch.autograd.backward(y)

    #print(f"{sals.shape=}")
    print(f"{sals=}")
    hess=torch.autograd.grad(sals,x) #, retain_graph=True)
    #hess=torch.autograd.grad(sals[0],x)
    print(f"{hess=}")
    exit()

    #""
    #print(f"{sals.mean()=}")
    for sal in sals:
        print(f"{sal=}")
        #hess=torch.autograd.grad(sal,x, retain_graph=True, allow_unused=True)[0]
        #hess=torch.autograd.grad(sal,x, retain_graph=True)[0]
        hess=torch.autograd.backward(sal,x, retain_graph=True)[0]
        print(f"{hess=}")
    exit()
    #""

    hessian=torch.zeros(x.shape[0],x.shape[1],x.shape[2],x.shape[1],x.shape[2])
    #""
    for i_batch in range(x.shape[0]):
        for i_alph in range(x.shape[1]):
            for i_seq in range(x.shape[2]):
                print(f"{sals[i_batch,i_alph,i_seq]=}")
                #hess=torch.autograd.grad(sals[i_batch,i_alph,i_seq],x, retain_graph=True)[0]
                #hess=torch.autograd.grad(sals[i_batch,i_alph,i_seq],x)[0]
                hess=torch.autograd.grad(sals[i_batch,i_alph,i_seq],x, retain_graph=True,allow_unused=True)[0]
                print(f"{hess.shape=}")
                for j_alph in range(x.shape[1]):
                    for j_seq in range(x.shape[2]):
                        hessian[i_batch,i_alph,i_seq,j_alph,j_seq]=hess[j_alph,j_seq]
    #""
    #print("HERE")
    return hessian

def backward_twice(y,x):
    #torch.autograd.gradcheck(y, x)
    #torch.autograd.gradgradcheck(y.apply, x)
    #x.retain_grad()
    #y.backward(create_graph=True)
    y.backward(create_graph=True, retain_graph=True) 
    #x.retain_grad()
    sal=x.grad.data.cpu().numpy()
    #x.retain_grad()
    y.retain_grad()
    print(f"{sal=}")
    ##y.zero_grad()
    y.backward()
    #y.backward(retain_graph=True) 
    #y.backward(create_graph=True, retain_graph=True)
    hess=x.grad.data.cpu().numpy()
    #x.zero_grad()
    print(f"{sal=}")
    print(f"{hess=}")
    return hess
    """
    #unc_all.mean().backward()
    unc_all.mean().backward(create_graph=True) # https://discuss.pytorch.org/t/how-to-backward-the-derivative/17662
    
    unc_sail = x.grad.data.cpu().numpy()
    batch_sails.append(unc_sail)
    print(f"{unc_sail.shape=} {unc_sail[0][0]=}")
    
    #unc_all.retain_grad() #maybe contained in create_graph?
    unc_all.mean().backward()
    unc_hess = x.grad.data.cpu().numpy()
    batch_hessians.append(unc_hess)
    print(f"{unc_hess.shape=} {unc_hess[0][0]=}")
    exit()
    """
    
if __name__=='__main__':
    torch.manual_seed(41)
    import torch.nn as nn

    #""
    class myNN(nn.Module):
        #def __init__(self, inpdim=1):
        def __init__(self, seq_len):
            super(myNN, self).__init__()
            #self.layer1=nn.Linear(seq_len,1)
            self.layer1=nn.Linear(seq_len,1) #,bias=False)
            self.layer2=nn.Linear(1,1) #,bias=False)

        def forward(self,x):
            out=x
            out=self.layer1(out)
            out=self.layer2(out)
            return out
    #""

    """
    class myNN(nn.Module):
        def __init__(self, alph, seq_len):
            super(myNN, self).__init__()
            self.flatten=nn.Flatten()
            self.layer1=nn.Linear(alph*seq_len,1) #,bias=False)
            #self.layer2=nn.Linear(1,1) #,bias=False)
            self.act=nn.Tanh()

        def forward(self,x):
            out=x
            out=self.flatten(out)
            #print(f"{out.shape=}")
            out=self.layer1(out)
            #out=self.layer2(out)
            out=self.act(out)
            return out
    """

    """
    alph=4
    #alph=2
    #seq_len=7         
    seq_len=3
    #x=torch.rand((10,4,seq_len)).requires_grad_(True)
    x=torch.rand((1,alph,seq_len)).requires_grad_(True)
    print(f"{x=}")
    mynn=myNN(alph, seq_len)
    print(mynn)
    print("derivative should be (a1,a2,...a6)")
    for parameter in mynn.parameters():
        print(parameter)
    print()
    """
    
    #""
    seq_len=1
    x=torch.rand((1,seq_len)).requires_grad_(True)
    mynn=myNN(seq_len=seq_len)
    #""

    #x=torch.rand(seq_len).requires_grad_(True)
    #y=nn.Sequential(nn.Flatten(), nn.Linear(4*seq_len,1))(x).mean()
    #y=nn.Sequential(nn.Flatten(), nn.Linear(seq_len,1))(x).mean()
    #y=nn.Sequential(nn.Linear(seq_len,1))(x)
    #y=nn.Linear(seq_len,1)(x)
    #y=nn.Sequential(nn.Linear(seq_len,1), nn.Linear(1,1))(x)
    
    #""
    y=mynn(x)
    #print(f"{mynn.parameters=}")
    """
    pars=[]
    for parameter in mynn.parameters():
        print(parameter) #,parameter.data)
        pars.append(parameter.data)
    #print(f"{pars[1]*(pars[0]*x)=}")
    print(f"{pars[2]*(pars[0]*x+pars[1])+pars[3]=}")
    print(f"derivative should be: {pars[0]*pars[2]=}")
    """

    #y=x.multiply(x.multiply(3))
    #y=(x.multiply(2).add(3)).multiply(2).add(3)
    #y=x.multiply(2*x).flatten().mean()
    #y.backward()

    print(f"{y=}")
    print()
    hess=calc_hessian_AC(y,x) #mean(0) will average the gradient per nt
    #hess=backward_twice(y,x)
    #jacob=torch.autograd.functional.jacobian(mynn,x)
    
    #hess=torch.autograd.functional.hessian(mynn,x)
    #print(f"{jacob=}")
    print(f"{hess=}")
    #print(f"{jacob.shape=}")
    print(f"{hess.shape=}")

    exit()

    # check symmetry
    for i_alph in range(alph):
        for i_seq in range(seq_len):
            #print(f"{i_alph=} {i_seq=} {hess[:,i_alph,i_seq,:,i_alph,i_seq]=}")
            print(f"{i_alph=} {i_seq=}")
            print(f"{hess[:,i_alph,i_seq,:,:,:].squeeze(0).squeeze(0)=}")
            print(f"{hess[:,:,:,:,i_alph,i_seq].squeeze(0).squeeze(-1)=}")
            #print(f"{hess[:,:,:,:,i_alph,i_seq].squeeze(0).squeeze(-1).shape=}")
            #indexes.append([i_alph,i_seq])
            #hessians.append(hess)
            print()  

    print("--- --- ---")

    def multidim_argsort(hess):
        flat=[]
        indexes=[]
        for i0 in range(hess.shape[0]):
            for i1 in range(hess.shape[1]):
                for i2 in range(hess.shape[2]):
                    for i3 in range(hess.shape[3]):
                        for i4 in range(hess.shape[4]):
                            for i5 in range(hess.shape[5]):
                                flat.append(hess[i0,i1,i2,i3,i4,i5])
                                indexes.append([i0,i1,i2,i3,i4,i5])
        return np.array(indexes)[np.argsort(flat)[::-1]] , flat

    indexes=[]
    hessians=[]    
    for i_seq in range(seq_len):
        for j_seq in range(seq_len):
            print(f"{i_seq=} {j_seq=} {hess[:,:,i_seq,:,:,j_seq].squeeze(2).squeeze(0)=}")
            #print(f"{i_seq=} {j_seq=} {hess[:,:,i_seq,:,:,j_seq].shape=}")
            indexes.append([i_alph,i_seq])

    #argsorts=torch.argsort(hess.flatten(),descending=True)
    argsorts,flat_hess=multidim_argsort(hess)
    print(f"{argsorts.shape=}")
    print(f"{argsorts[0]=}")
    print(f"{hess[argsorts[0][0],argsorts[0][1],argsorts[0][2],argsorts[0][3],argsorts[0][4],argsorts[0][5]]=}")
    print(f"{hess[argsorts[-1][0],argsorts[-1][1],argsorts[-1][2],argsorts[-1][3],argsorts[-1][4],argsorts[-1][5]]=}")

    #print("PROSS: FARE DOPPIO BACKWARD CON UNA SEMPLICE DOPPIA NN.LINEAR E VEDERE SE TI TORNA, MAGARI USANDO UNA ACTIVATION FUNCTION ")