# https://github.com/niralisomia/Diffusion_Small_Data/blob/main/seq_evals_improved.py
import numpy as np
import scipy.linalg as linalg
import torch

class EmbeddingExtractor:
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        self.embedding = output.detach()

def get_penultimate_embeddings(model, x):
    # Find the penultimate layer
    extractor = EmbeddingExtractor()
    """
    model
    model.stem
    model.stem.block
    model.stem.block.0
    model.stem.block.1
    model.stem.block.2
    model.main
    model.main.0
    model.main.0.0
    model.main.0.0.fn
    model.main.0.0.fn.block
    model.main.0.0.fn.block.0
    model.main.0.0.fn.block.1
    model.main.0.0.fn.block.2
    model.main.0.0.fn.block.3
    model.main.0.0.fn.block.4
    model.main.0.0.fn.block.5
    model.main.0.0.fn.block.6
    model.main.0.0.fn.block.6.fc
    model.main.0.0.fn.block.6.fc.0
    model.main.0.0.fn.block.6.fc.1
    model.main.0.0.fn.block.6.fc.2
    model.main.0.0.fn.block.6.fc.3
    model.main.0.0.fn.block.7
    model.main.0.0.fn.block.8
    model.main.0.0.fn.block.9
    model.main.0.1
    model.main.0.1.block
    model.main.0.1.block.0
    model.main.0.1.block.1
    model.main.0.1.block.2
    model.main.0.2
    model.main.0.3
    model.main.1
    model.main.1.0
    model.main.1.0.fn
    model.main.1.0.fn.block
    model.main.1.0.fn.block.0
    model.main.1.0.fn.block.1
    model.main.1.0.fn.block.2
    model.main.1.0.fn.block.3
    model.main.1.0.fn.block.4
    model.main.1.0.fn.block.5
    model.main.1.0.fn.block.6
    model.main.1.0.fn.block.6.fc
    model.main.1.0.fn.block.6.fc.0
    model.main.1.0.fn.block.6.fc.1
    model.main.1.0.fn.block.6.fc.2
    model.main.1.0.fn.block.6.fc.3
    model.main.1.0.fn.block.7
    model.main.1.0.fn.block.8
    model.main.1.0.fn.block.9
    model.main.1.1
    model.main.1.1.block
    model.main.1.1.block.0
    model.main.1.1.block.1
    model.main.1.1.block.2
    model.main.1.2
    model.main.1.3
    model.main.2
    model.main.2.0
    model.main.2.0.fn
    model.main.2.0.fn.block
    model.main.2.0.fn.block.0
    model.main.2.0.fn.block.1
    model.main.2.0.fn.block.2
    model.main.2.0.fn.block.3
    model.main.2.0.fn.block.4
    model.main.2.0.fn.block.5
    model.main.2.0.fn.block.6
    model.main.2.0.fn.block.6.fc
    model.main.2.0.fn.block.6.fc.0
    model.main.2.0.fn.block.6.fc.1
    model.main.2.0.fn.block.6.fc.2
    model.main.2.0.fn.block.6.fc.3
    model.main.2.0.fn.block.7
    model.main.2.0.fn.block.8
    model.main.2.0.fn.block.9
    model.main.2.1
    model.main.2.1.block
    model.main.2.1.block.0
    model.main.2.1.block.1
    model.main.2.1.block.2
    model.main.2.2
    model.main.2.3
    model.main.3
    model.main.3.0
    model.main.3.0.fn
    model.main.3.0.fn.block
    model.main.3.0.fn.block.0
    model.main.3.0.fn.block.1
    model.main.3.0.fn.block.2
    model.main.3.0.fn.block.3
    model.main.3.0.fn.block.4
    model.main.3.0.fn.block.5
    model.main.3.0.fn.block.6
    model.main.3.0.fn.block.6.fc
    model.main.3.0.fn.block.6.fc.0
    model.main.3.0.fn.block.6.fc.1
    model.main.3.0.fn.block.6.fc.2
    model.main.3.0.fn.block.6.fc.3
    model.main.3.0.fn.block.7
    model.main.3.0.fn.block.8
    model.main.3.0.fn.block.9
    model.main.3.1
    model.main.3.1.block
    model.main.3.1.block.0
    model.main.3.1.block.1
    model.main.3.1.block.2
    model.main.3.2
    model.main.3.3
    model.mapper
    model.mapper.block
    model.mapper.block.0
    model.mapper.block.1
    model.mapper.block.2
    model.head
    model.head.0
    model.head.1
    model.head.2
    model.head.3
    model.head.4

    self.head = nn.Sequential(nn.Linear(out_ch * 2, out_ch * 2),
                                nn.Dropout(0.5), #AC
                            nn.BatchNorm1d(out_ch * 2),
                            activation(),
                            loss_for_evidential.DenseNormalGamma(out_ch * 2, 1))
    """
    if model.name=='DeepSTARR':
        layername='model.batchnorm6'
    if model.name=='LegNet':
        #layername='model.head.4'
        #layername='model.stem.block.2'
        #layername='model.mapper.block.2'
        layername='model.head.0'
    for name, module in model.named_modules():
        if name==layername:
            handle = module.register_forward_hook(extractor.hook)
            break
    else:
        raise ValueError("Could not find layer "+layername)

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove the hook
    handle.remove()

    return extractor.embedding

def calculate_activation_statistics(embeddings):
    #embeddings_d = embeddings.detach().numpy()
    embeddings_d = embeddings.detach().cpu().numpy()
    mu = np.mean(embeddings_d, axis=0)
    sigma = np.cov(embeddings_d, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    #Frechet distance: d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        #if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3): # AC this should be: chekcing if all elements are withing a toleramce to a value
        #    m = np.max(np.abs(covmean.imag))
        #    raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_frechet_distance(model,X1,X2):
    embeddings1 = get_penultimate_embeddings(model, X1)
    embeddings2 = get_penultimate_embeddings(model, X2)
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    #print(f"{mu1=} {sigma1=} {mu2=} {sigma2=}")
    #print(f"{sigma1=} {(0. in sigma1)=}")remove_sigmas_that_are_0_or_close_to_zero
    print(f"{mu1.shape=} {sigma1.shape=} {mu2.shape=} {sigma2.shape=}")
    frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return frechet_distance

if __name__=='__main__':
    import h5py
    import torch
    from PL_Models import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    which='K562'
    chosen_model='LegNetPK'
    #chosen_model='DeepSTARR'

    if chosen_model=='LegNetPK': input_h5_file='../inputs/newLentiMPRA'+which+'_processed_for_dal.h5'
    if chosen_model=='DeepSTARR': input_h5_file='../inputs/DeepSTARRdev.h5'

    #ckptf=mydir+outflag+'/ckpt_'+str(model_index)+"_"+str(index_from)+"_ial-"+str(i_al)+".ckpt"
    if chosen_model=='LegNetPK': ckptf='../inputs/oracle_LegNetPK_newLentiMPRAK562_processed_for_dal_finetune.ckpt'
    if chosen_model=='DeepSTARR': ckptf='../inputs/oracle_DeepSTARR_DeepSTARRdev_1.ckpt'
    #if use_oracle_grad: ckptf='../inputs/oracle_ResidualBind_LentiMPRA_processed_for_dal_relustandard1.ckpt'

    #####

    data=h5py.File(input_h5_file,'r')
    #X1=torch.tensor(np.array(data['X_train']))
    X1=torch.tensor(np.array(data['X_train']))[:10]
    #X1=torch.tensor(np.array(data['X_train']))[0].unsqueeze(0)
    #X2=torch.tensor(np.array(data['X_train']))[10:20]
    #X2=torch.tensor(np.array(data['X_test']))
    #X2=torch.tensor(np.array(data['X_test']))[:10]
    #X2=torch.tensor(np.array(data['X_test']))[0].unsqueeze(0)
    X2=X1
    #X1=X1.to(device)
    #X2=X2.to(device)

    model=eval("PL_"+chosen_model+"(input_h5_file='"+input_h5_file+"', initial_ds=True)") #, extra_str='"+extra_str+"')") # QUIQUIURG doesnt change anything in terms of pred? it only counts the ckpt you use??
        
    model = model.load_from_checkpoint(ckptf, input_h5_file=input_h5_file)
    model.eval()
    ##deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path).eval()

    #exit()
    
    frech=get_frechet_distance(model,X1,X2)
    print(frech)