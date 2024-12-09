import torch
import numpy as np
import tqdm
import nlist_sorting
from matplotlib import pyplot as plt
from scipy.stats import gumbel_r
import differentiables_AC


def turn_y_into_prob(y, keepgrad = True):
    prob=y
    print("WORK IN PROGRESS: turn_y_into_prob")
    exit()
    return prob

def model_positionwise_mean(logits, keepgrad = True):
    return(torch.mean(logits, dim=0))
def mutual_inform(logits, keepgrad = True):
    mean_prob = model_positionwise_mean(logits, keepgrad = keepgrad) # Positionwise entropy, average probability
    log_mean_prob = torch.log2(mean_prob)
    # calculate entropy
    pmeanlogpmean = torch.multiply(mean_prob,log_mean_prob)
    predictive_entropy = torch.multiply(torch.sum(pmeanlogpmean, dim=-2), -1)
    #Conditional entropy of positions over weights (approximated)
    log_p = torch.log2(logits)
    plogp = torch.multiply(logits, log_p)
    position_entropy = torch.sum(torch.multiply(plogp, -1), dim=-2)
    conditional_entropy = torch.mean(position_entropy, dim=0)
    mutual_info = torch.subtract(predictive_entropy,conditional_entropy)
    return(torch.mean(mutual_info, dim=-1))


def single_true(iterable):
    """
    This function takes an iterable and returns true only if exactly one value in the iterable is true.
    It relies on the fact that any consumes an iterable only until it hits a true value.
    
    Parameters
    ----------
    iterable: an iterable
    
    returns
    -------
    bool: True if exactly one element is True
    """
    
    i = iter(iterable)
    return any(i) and not any(i)

def plot_distrib_with_threshold(data,threshold, flag='_',nbins=100, outdir='./',color='C0'):
    """ 
    Plot histogram of a distribution. 
    
    Parameters
    ----------
    data: np.array points from the distribution
    threshold: float a threshold to draw
    flag: string added to the file name of the saved figure
    nbins: int passed to plt.hist
    outdir: string directory to save output
    color: passed to plt.hist
    
    returns
    -------
    None
    """
    fig = plt.figure(1, figsize=(5, 4))
    plt.hist(data, bins=nbins, density=False,color=color)
    plt.axvline(x=threshold,color='red') # vertical line
    outfile=outdir+'Hist_'+flag+'.png'
    plt.xlim(0.0,1.5)
    fig.savefig(outfile,dpi=150)
    print("DONE: "+outfile)
    plt.clf()
    
def top_k_select(scores, k):
    return(np.argsort(scores)[::-1][:k])

def random_selection(scores, k):
    return(np.random.choice(range(len(scores)),size=k))

def softmax_select(scores, k, beta, seed = None):
    if beta == 0.0:
        idx = random_selection(scores, k)
    else:
        jittered = scores + gumbel_r.rvs(loc=0, scale=1 / beta, size=len(scores), random_state=seed)
        idx = random_selection(jittered, k, beta)
        
    return(idx)

def power_select(scores, k, beta, seed = None):
    assert (scores>=0).all(), 'Cannot perform power selection with negative scores'
    scores =  np.log(scores)
    idx = softmax_select(scores, k, beta, seed = seed)
    return(idx)

def softrank_select(scores, k, beta, seed = None):
    sort_idx = np.argsort(-1*np.array(scores))
    ranks = np.argsort(sort_idx)
    idx = power_select(1 / ranks, beta, k, seed = seed)
    return(idx)
    


class Ranker():
    def __init__(self,
                 Predictive_Models,
                                 batch_size,how_many_batches=1,
                                 #rank_method='entropy',
                                 uncertainty_method='no',diversity_method='no',highpred_method='no',
                                 uncertainty_weight=0.0,diversity_weight=0.0,highpred_weight=0.0,
                                 chosen_model='InHouseCNN',
                                 cycle=1,sigmadistr_freq=1,outdir='./',outflag='_',device='cuda', local_seed=41, task_type='single_task_binary_classification',
                                 ):
        self.Predictive_Models=Predictive_Models
        self.batch_size=batch_size
        self.how_many_batches=how_many_batches
        
        self.uncertainty_method=uncertainty_method
        self.uncertainty_weight=uncertainty_weight
        self.diversity_method=diversity_method
        self.diversity_weight=diversity_weight
        self.highpred_method=highpred_method
        self.highpred_weight=highpred_weight
        
        """
        if self.uncertainty_method=='no':
            assert  self.uncertainty_weight == 0, 'An uncertainty weight was passed without an uncertainty method'
        if self.diversity_method=='no':
            assert  self.diversity_weight == 0, 'An diversity weight was passed without an diversity method'
        if self.highpred_method=='no':
            assert  self.highpred_weight == 0, 'An highpred weight was passed without an highpred method'
        """
        
        self.chosen_model=chosen_model,
        self.cycle=cycle
        self.sigmadistr_freq=sigmadistr_freq
        self.outdir=outdir
        self.outflag=outflag
        self.device=device
        self.local_seed=local_seed 
        self.task_type=task_type

    def calculate_desideratum(self, pred, j, x_batch, ###y_batch, #EL2Nold
                              keep_grads = False):
        if 'mc_dropout' in self.uncertainty_method: 
            #print("UNCALLDEBUG")
            n_mc=int(self.uncertainty_method.replace('mc_dropout_','')) #QUIQUINONURG should this become a separate args?
            preds_mc=torch.zeros((n_mc,len(x_batch))).to(self.device) #QUIQUINONURG torch.zeros better than torch.empty?
            for j_mc in range(n_mc):     
                preds_mc[j_mc]=self.Predictive_Models[j].to(self.device).predict_custom_mcdropout(x_batch,seed=self.local_seed+j_mc*1000, keepgrad = keep_grads).squeeze(axis=1).unsqueeze(axis=0)#.detach().cpu() #it's indeed Predictive_Models[j], NOT Predictive_Models[j_mc]
                #print(f"UNCALLDEBUG: {preds_mc[j_mc]=} {self.local_seed+j_mc*1000=}")
            des=torch.std(preds_mc,axis=0)
            #print(f"UNCALLDEBUG: {des=}")
        if self.uncertainty_method=='one_over_margin': 
            des=torch.divide(1,torch.multiply(pred,-1).add(1).multiply(-1).add(pred).pow(2).sqrt()) # 1/abs(x-(1-x)) 
        if 'entropy' in self.uncertainty_method: 
            des=differentiables_AC.AC_binary_entropy(pred.squeeze(axis=1),self.device)
        if self.uncertainty_method=='sigma_deep_ensemble': 
            #print(f"for debugging sigma_deep_ensembe: {len(self.Predictive_Models)=}")
            des=torch.zeros(x_batch.shape[0]) # dummy
            #print("ERROR: WORK IN PROGRESS: calculate desideratum in ranker.py for sigma_deep_ensemble - I just wrote the lines just below, are they ok? I think so")
            preds_deepens=torch.zeros((len(self.Predictive_Models),len(x_batch))).to(self.device)
            for jj in range(len(self.Predictive_Models)): # j is dummy, so I'll use jj here instead (QUIQUIURG is this ok?)     
                #preds_deepens[jj]=self.Predictive_Models[j].to(self.device).predict_custom(x_batch, keepgrad = keep_grads).squeeze(axis=1).unsqueeze(axis=0)#.detach().cpu() #it's indeed Predictive_Models[j], NOT Predictive_Models[j_mc]
                preds_deepens[jj]=self.Predictive_Models[jj].to(self.device).predict_custom(x_batch, keepgrad = keep_grads).squeeze(axis=1).unsqueeze(axis=0)#.detach().cpu() #it's indeed Predictive_Models[j], NOT Predictive_Models[j_mc]
            des=torch.std(preds_deepens,axis=0)
            #exit()
        if self.uncertainty_method=='EL2N':  #EL2N # https://colab.research.google.com/drive/13RY_7eEYRhZvsAR1fDfrbCEf-bjnO2vV?usp=sharing
            #des=torch.zeros(x_batch.shape[0]) # dummy
            
            ##preds = self.Predictive_Models[j].predict_custom(x_batch, keepgrad = keep_grads) 
            ##y_true= PROX: follow the trail of x_batch: you must use Unlabelled_Y in rank()
            """             
            abs_diff = np.abs(predictions-y_train_np)
            el2n_final = np.mean(abs_diff,axis=1,keepdims=True)

            el2n_file_path = '/content/drive/MyDrive/Trial/el2n.csv'
            el2n_scores = np.genfromtxt(el2n_file_path, delimiter=',')

            threshold_percent = 10
            threshold = np.percentile(el2n_scores, threshold_percent)
            indices_below_threshold = np.where(el2n_scores < threshold)[0]

            ### EL2N:epochs-to-ALcycle?
            """
            ##des=self.Predictive_Models[j].get_el2n_scores() # this wouldn't work bc the model, in part II, will have the weights, but the list of el2n scores per epoch will be again [] #EL2N
            des=self.Predictive_Models[j].get_el2n_scores_with_load(self.outdir+'el2n-per-epoch.npy')  #EL2N
            print(f"4 el2n: {des.shape=}")
        if self.uncertainty_method=='BALD':
            preds=self.Predictive_Models[j].to(self.device).predict_custom(x_batch, keepgrad = keep_grads).squeeze(axis=1).unsqueeze(axis=0)
            if 'regr' in self.task_type:
                prob=turn_y_into_prob(preds, keepgrad = keep_grads)
            else:
                prob=preds
            des=mutual_inform(prob, keepgrad = keep_grads)
        if self.uncertainty_method=='BatchBALD':
            preds=self.Predictive_Models[j].to(self.device).predict_custom(x_batch, keepgrad = keep_grads).squeeze(axis=1).unsqueeze(axis=0)
            if 'regr' in self.task_type:
                prob=turn_y_into_prob(preds, keepgrad = keep_grads)
            else:
                prob=preds
            print("WORK IN PROGRESS BatchBALD")
            exit()
        #print("UNCALLDEBUG:",des)
        return des
    
    def get_pred(self, Unlabelled_X, keep_grads = False):
        X_loader=torch.utils.data.DataLoader(Unlabelled_X, batch_size=self.batch_size, shuffle=False)
        if keep_grads:
            allj_preds=torch.empty(0).to(self.device)
        else:
            allj_preds=torch.empty(0)
        #for j in tqdm.tqdm(range(len(self.Predictive_Models)), total=len(self.Predictive_Models), desc='Making Prediction over U with '+str(len(self.Predictive_Models))+' models', colour='blue'): #QUIQUIURG is this to be parallelized too?
        for j in range(len(self.Predictive_Models)): #QUIQUIURG is this to be parallelized too?
            self.Predictive_Models[j].to(self.device).eval()
            if keep_grads:
                allbatches_preds=torch.empty(0).to(self.device)
            else:
                allbatches_preds=torch.empty(0).to('cpu')
            for i_batch,x_batch in enumerate(X_loader):
                x_batch = x_batch.to(self.device)
                pred = self.Predictive_Models[j].predict_custom(x_batch,
                                                                keepgrad = keep_grads) 
                print(f"ALEATDEBUG {pred.shape=}")
                #print(f"ALEATDEBUG {self.Predictive_Models[j].model.has_aleatoric=}")
                if not keep_grads:
                    pred=pred.detach().to('cpu')
                pred_sq=pred.squeeze(axis=1) 
                allbatches_preds=torch.cat((allbatches_preds,pred_sq),axis=0)
            allj_preds=torch.cat((allj_preds,allbatches_preds.unsqueeze(0)),axis=0)
        print(f"ALEATDEBUG: {allj_preds.shape=}")
        preds_av=allj_preds.mean(axis=0) #average across models
        return preds_av
    
    def pred_unc_evidential(self, Unlabelled_X, keep_grads = False):
        """
        mu, logv, alpha, beta=self.get_pred(Unlabelled_X, keep_grads)
        ValueError: too many values to unpack (expected 4)
        """
        """
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1) # continuous.py
        """
        evidential_output=self.get_pred(Unlabelled_X, keep_grads)
        print(f"EVIDDEBUG {len(evidential_output)=}")
        gamma=evidential_output[0]
        nu=evidential_output[1]
        alpha=evidential_output[2]
        beta=evidential_output[3]
        preds_av=gamma
        aleat_uncs=beta/(alpha-1)
        epist_uncs=beta/(nu*(alpha-1))
        return preds_av, aleat_uncs, epist_uncs 
    
    def calculate_desiderata(self, Unlabelled_X, keep_grads = False):
        X_loader=torch.utils.data.DataLoader(Unlabelled_X, batch_size=self.batch_size, shuffle=False)
        #Xy_loader=torch.utils.data.DataLoader(list(zip(Unlabelled_X, Unlabelled_Y)), batch_size=self.batch_size, shuffle=False) #EL2N
        if keep_grads:
            allj_preds=torch.empty(0).to(self.device)
            allj_des_s=torch.empty(0).to(self.device)
        else:
            allj_preds=torch.empty(0)
            allj_des_s=torch.empty(0)
        #for j in tqdm.tqdm(range(len(self.Predictive_Models)), total=len(self.Predictive_Models), desc='Making Prediction over U with '+str(len(self.Predictive_Models))+' models', colour='blue'): #QUIQUIURG is this to be parallelized too?
        for j in range(len(self.Predictive_Models)): #QUIQUIURG is this to be parallelized too?
            self.Predictive_Models[j].to(self.device).eval()
            if keep_grads:
                allbatches_preds=torch.empty(0).to(self.device)
                allbatches_des_s=torch.empty(0).to(self.device)
            else:
                allbatches_preds=torch.empty(0).to('cpu')
                allbatches_des_s=torch.empty(0).to('cpu')
            for i_batch,x_batch in enumerate(X_loader):
            #for i_batch,batch in enumerate(Xy_loader): #EL2N
                #x_batch=batch[0] #EL2N
                #y_batch=batch[1] #EL2N
                x_batch = x_batch.to(self.device)
                #y_batch = y_batch.to(self.device) #EL2N
                pred = self.Predictive_Models[j].predict_custom(x_batch,
                                                                keepgrad = keep_grads) 
                #print(f"ALEATDEBUG {pred.shape=}")
                #print(f"ALEATDEBUG {self.Predictive_Models[j].model.has_aleatoric=}")
                if not keep_grads:
                    pred=pred.detach().to('cpu')
                des_value=self.calculate_desideratum(pred, j, 
                                                     x_batch, #y_batch, #EL2N 
                                                     keep_grads = keep_grads)
                if not keep_grads:
                    des_value=des_value.detach().to('cpu')
                pred_sq=pred.squeeze(axis=1) 
                allbatches_preds=torch.cat((allbatches_preds,pred_sq),axis=0)
                allbatches_des_s=torch.cat((allbatches_des_s,des_value),axis=0)

            allj_preds=torch.cat((allj_preds,allbatches_preds.unsqueeze(0)),axis=0)
            allj_des_s=torch.cat((allj_des_s,allbatches_des_s.unsqueeze(0)),axis=0)
        #print('allj_preds: ',allj_preds)
        #print('allj_preds.shape: ',allj_preds.shape)
        #print(f"ALEATDEBUG: {allj_preds.shape=}")
        preds_av=allj_preds.mean(axis=0) #average across models
#         print('preds_av: ',preds_av)
        preds_standard_devs=allj_preds.std(axis=0)
#         print('preds_standard_devs: ',preds_standard_devs)
        if (preds_standard_devs==0).any():
            print(f"NANCHECK: std is zero.")

        des_avs=allj_des_s.mean(axis=0)
#         print('des_avs: ',des_avs)
        #des_stds=allj_des_s.std(axis=0)
#         print('des_stds: ',des_stds)

        if self.uncertainty_method=='sigma_deep_ensemble': 
            desiderata=preds_standard_devs 
        #else:
        #    desiderata=des_avs

        # QUIQUINONURG ADD ALL THE MISSING ONES
        #allj_preds=allj_preds.detach().cpu() #.to('cpu') #QUIQUINONURG
        #preds_standard_devs=preds_standard_devs #.detach().cpu() #.to('cpu') #QUIQUINONURG
        elif 'entropy' in self.uncertainty_method: 
            desiderata=des_avs #preds_entropies_devs #.detach().cpu() #.to('cpu') #QUIQUINONURG
        elif 'mc_dropout' in self.uncertainty_method: 
            desiderata=des_avs #.detach().cpu() #.to('cpu') #QUIQUINONURG
#         print('desiderata: ',desiderata)
        return desiderata,preds_av

    def calculate_desiderata_4Hess(self, Unlabelled_X): 
        ##keep_grads=False
        keep_grads=True
        X_loader=torch.utils.data.DataLoader(Unlabelled_X, batch_size=self.batch_size, shuffle=False)
        if keep_grads:
            allj_preds=torch.empty(0).to(self.device)
            allj_des_s=torch.empty(0).to(self.device)
        else:
            allj_preds=torch.empty(0)
            allj_des_s=torch.empty(0)
        #for j in tqdm.tqdm(range(len(self.Predictive_Models)), total=len(self.Predictive_Models), desc='Making Prediction over U with '+str(len(self.Predictive_Models))+' models', colour='blue'): #QUIQUIURG is this to be parallelized too?
        for j in range(len(self.Predictive_Models)): #QUIQUIURG is this to be parallelized too?
            self.Predictive_Models[j].to(self.device).eval()
            if keep_grads:
                allbatches_preds=torch.empty(0).to(self.device)
                allbatches_des_s=torch.empty(0).to(self.device)
            else:
                allbatches_preds=torch.empty(0).to('cpu')
                allbatches_des_s=torch.empty(0).to('cpu')
            for i_batch,x_batch in enumerate(X_loader):
                x_batch = x_batch.to(self.device)
                pred = self.Predictive_Models[j].predict_custom(x_batch,
                                                                keepgrad = keep_grads) 
                if not keep_grads:
                    pred=pred.detach().to('cpu')
                des_value=self.calculate_desideratum(pred, j, 
                                                     x_batch, 
                                                     keep_grads = keep_grads)
                if not keep_grads:
                    des_value=des_value.detach().to('cpu')
                pred_sq=pred.squeeze(axis=1) 
                allbatches_preds=torch.cat((allbatches_preds,pred_sq),axis=0)
                allbatches_des_s=torch.cat((allbatches_des_s,des_value),axis=0)

            allj_preds=torch.cat((allj_preds,allbatches_preds.unsqueeze(0)),axis=0)
            allj_des_s=torch.cat((allj_des_s,allbatches_des_s.unsqueeze(0)),axis=0)
#         print('allj_preds: ',allj_preds)
#         print('allj_preds.shape: ',allj_preds.shape)
        preds_av=allj_preds.mean(axis=0)
#         print('preds_av: ',preds_av)
        preds_standard_devs=allj_preds.std(axis=0)
#         print('preds_standard_devs: ',preds_standard_devs)

        des_avs=allj_des_s.mean(axis=0)
#         print('des_avs: ',des_avs)
        #des_stds=allj_des_s.std(axis=0)
#         print('des_stds: ',des_stds)

        if self.uncertainty_method=='sigma_deep_ensemble': 
            desiderata=preds_standard_devs 
        #else:
        #    desiderata=des_avs

        # QUIQUINONURG ADD ALL THE MISSING ONES
        #allj_preds=allj_preds.detach().cpu() #.to('cpu') #QUIQUINONURG
        #preds_standard_devs=preds_standard_devs #.detach().cpu() #.to('cpu') #QUIQUINONURG
        elif 'entropy' in self.uncertainty_method: 
            desiderata=des_avs #preds_entropies_devs #.detach().cpu() #.to('cpu') #QUIQUINONURG
        elif 'mc_dropout' in self.uncertainty_method: 
            desiderata=des_avs #.detach().cpu() #.to('cpu') #QUIQUINONURG
#         print('desiderata: ',desiderata)
        return desiderata

    def combine_n_lists(self, arrays, weights, normalize=True):
        assert sum(weights) == 1, "Weights must sum to 1"
        s1=np.zeros(len(arrays[0]))
        ref_len=len(arrays[0])
        for i_a,arr in enumerate(arrays):
            assert len(arr)==ref_len, "Error: two lists for nlist-sorting are not of same length: " + str(i_a)
            if normalize:
                arrsum=np.sum(arr)
                if arrsum!=0.0:
                    norm_arr=len(arrays[0])*arr/np.sum(arr) #QUIQUIURG is this really allowing for a comparison of lists with different maximums?
                else:
                    if (arr==0.0).all():
                        norm_arr=arr 
                    else:
                        norm_arr=arr #QUIQUIURG probably this is not ok in a situation where not everything is strictly 0 #JJD: consider switching to z-score?
            else:
                norm_arr=arr
            #print(f"{weights[i_a]=} {norm_arr=}")
            #s1+=weights[i_a]*norm_arr
            for i_s in range(len(s1)): #     s1+=weights[i_a]*norm_arr : TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please do not rely on this error; it may not be given on all Python implementations.
                s1[i_s]+=weights[i_a]*norm_arr[i_s]
            #print(f"{weights[i_a]=} {arrsum=} {(arr==0.0).all()=}")
        return s1
    
    def select_batch(self, scores, k):
        return(top_k_select(scores,k))
    
    def rank(self, Unlabelled_X, Unlabelled_Y):
        # # # Ranking of the sequences # # #
        method_arr = [self.uncertainty_method!='no', self.diversity_method!='no', self.highpred_method!='no']
            
        if any(method_arr):
            unc_scores,preds_av=self.calculate_desiderata(Unlabelled_X, keep_grads = False) 
            #print(f"CONTROL {unc_scores.shape=}")
            if self.uncertainty_method=='no': unc_scores=np.zeros(len(Unlabelled_X))
            assert not torch.isnan(unc_scores).any(), "ERROR: at least one uncertainty is NaN."

            # # # Diversity
        
            if self.diversity_method=='no':
                div_scores=np.zeros(len(Unlabelled_X))
            """
            elif  self.diversity_method=='kmer_diversity': #QUIQUIURG this is implementing novelty, not diversity, as of now!!!
                min_k=3 #2 #QUIQUIURG will have to be: int(rank_method.replace('mc_dropout_',''))
                max_k=6 #4
                reference_kmer_distrib=kmer_freq_KL.kmer_frequencies_AC(Predictive_Models[0].X_test, min_k=min_k,max_k=max_k) # Predictive_Models[0] is ok, since its X_test is the same of orig_data_module!
                KLdivs=kmer_freq_KL.KL_divergences(Unlabelled_X,reference_kmer_distrib,min_k=min_k,max_k=max_k)
                div_scores=KLdivs
            elif self.diversity_method=='y_diversity':
                div_scores=nlist_sorting.diversity_scores(preds_av,npoints=batch_size) #QUIQUIURG are preds_av logits? so is this method equally applying to binary classification and to regression?
            """

            # # # High prediction scores

            if self.highpred_method=='no':
                highpred_scores=np.zeros(len(Unlabelled_X))
            elif self.highpred_method=='yes':
                highpred_scores=preds_av #.detach().cpu().numpy()
            
            final_scores=self.combine_n_lists(arrays=[unc_scores, div_scores, highpred_scores], weights=[self.uncertainty_weight, self.diversity_weight, self.highpred_weight], normalize = not single_true(method_arr))
            #provaprova=len(final_scores)*final_scores/np.sum(final_scores)
            #print(f"{provaprova[:10]=} {unc_scores[:10]}") # LAST ONE: they are not the same, even if they are expected to be, e.g.: 0.012 vs 0.014: probably I was dividing by too large of a number, when normalizing, but even multiplying by the len is useless

            """ # This should be ok to remove by using normalize = not single_true
            if self.uncertainty_weight==1.0 and self.diversity_weight==0.0 and self.highpred_weight==0.0: #QUIQUIURG this section prevents to see if n_list_sorting behaves correctly 
                #new_batch_indexes=np.argsort(unc_scores)
                final_scores=unc_scores
            if self.uncertainty_weight==0.0 and self.diversity_weight==1.0 and self.highpred_weight==0.0:
                #new_batch_indexes=np.argsort(div_scores)
                final_scores=div_scores
            if self.uncertainty_weight==0.0 and self.diversity_weight==0.0 and self.highpred_weight==1.0:
                #new_batch_indexes=np.argsort(highpred_scores)
                final_scores=highpred_scores
            """

            #print(f"A2CHEKCrank {len(self.how_many_batches)=} {len(self.batch_size)=}")
            new_batch_indexes_batched=list(self.select_batch(final_scores,k=self.how_many_batches*self.batch_size))
            #print(f"A2CHEKCrank {len(new_batch_indexes_batched)=}")

            #cum_perc_unc=np.sum(np.sort(unc_scores)[new_batch_indexes_batched])/np.sum(unc_scores) #QUIQUINONURG useless formula to monitor as it is now? Since I was using it to check that the model would get less and less uncertain. SOLUTION: use unc_thrs only
            #assert not np.isnan(cum_perc_unc), f'cum_perc_unc is nan\n{np.sum(np.sort(unc_scores)[new_batch_indexes_batched])=}\n{np.sum(unc_scores)=}'
            if len(Unlabelled_X)>0: 
                threshold=unc_scores[new_batch_indexes_batched].min()
            else:
                threshold=0.0

            """
            # QUIQUIURG reinsert this section. Last error:
            # ValueError: The 'color' keyword argument must have one color per dataset, but 11396 datasets and 1 colors were provided
            # However:  unc_scores.shape=torch.Size([11396]) just as I would expect.
            if self.cycle<self.sigmadistr_freq or self.cycle%self.sigmadistr_freq==0: 
                plot_distrib_with_threshold(unc_scores, threshold, flag=self.outflag+'_cycle-'+str(self.cycle),nbins=100, outdir=self.outdir, color='C0')
                if self.task_type=='single_task_binary_classification':
                    plot_distrib_with_threshold(unc_scores[Unlabelled_Y<=0.5], threshold, flag=self.outflag+'_cycle-'+str(self.cycle)+'_class0',nbins=100, outdir=self.outdir, color='C1')
                    plot_distrib_with_threshold(unc_scores[Unlabelled_Y>0.5], threshold, flag=self.outflag+'_cycle-'+str(self.cycle)+'_class1',nbins=100, outdir=self.outdir, color='C2')
            """
        else:
            #new_batch_indexes_batched=list(random_selection(errors,self.how_many_batches*self.batch_size))
            new_batch_indexes_batched=list(random_selection(np.arange(len(Unlabelled_X)),self.how_many_batches*self.batch_size))
            threshold=0.0
        
        cum_perc_unc=0.0 #dummy QUIQUINONURG
        return new_batch_indexes_batched,cum_perc_unc,threshold

    
class PowerRanker(Ranker):
    def __init__(self, *args, beta=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def select_batch(self, scores, k):
        return(power_select(scores,k, self.beta, seed=self.seed))
    
class SoftmaxRanker(Ranker):
    def __init__(self, *args, beta=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def select_batch(self, scores, k):
        return(softmax_select(scores,k, self.beta, seed=self.seed))
    
class SoftrankRanker(Ranker):
    def __init__(self, *args, beta=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def select_batch(self, scores, k):
        return(softrank_select(scores,k, self.beta, seed=self.seed))
    
    

# if __name__ == '__main__':
#     print('Running some ranker tests')
#     from PL_Models import PL_InHouseCNN
    
#     pairlist=[
#               ['InHouseCNN', 'ATF2_200'], 
#               #['InHouseCNN', 'TIA1_K562_200'],  

#               #['InHouseCNN', 'RBFOX2_K562_200'],
#               #['InHouseCNN', 'HNRNPK_K562_200'], 
#               #['InHouseCNN', 'PTBP1_K562_200'],
#               #['InHouseCNN', 'PUM2_K562_200'],  
#               #['InHouseCNN', 'QKI_K562_200'], 
#               #['InHouseCNN', 'SF3B4_K562_200'], 
#               #['InHouseCNN', 'SRSF1_K562_200'],  
#               #['InHouseCNN', 'TARDBP_K562_200'],  
#               #['InHouseCNN', 'U2AF1_K562_200'],

#               #['InHouseCNN', 'pseudolabels_ATF2_200'], #from PL_Models_infere.py
              
#               #['DeepSTARR', 'DeepSTARRdev'],
#               #['DeepSTARR', 'DeepSTARRhk'], #QUIQUIURG this has higher PCC! 0.75!
#     ] 

#     for pair in pairlist:
                
#         chosen_model=pair[0]
#         chosen_dataset=pair[1]

#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         device = 'cpu'
        
#         overall_seed=41
#         myseed=overall_seed
#         torch.manual_seed(myseed)
#         random.seed(myseed)
#         np.random.seed(myseed)
        
#         print("Setting model...")
#         if chosen_model=='InHouseCNN': 
#             model=PL_InHouseCNN(input_h5_file='../inputs/'+chosen_dataset+'.h5', initial_ds=True)
#             task_type = 'single_task_binary_classification'
#         if chosen_model=='DeepSTARR': 
#             model=PL_DeepSTARR(input_h5_file='../inputs/'+chosen_dataset+'.h5', initial_ds=True)
#             task_type = 'single_task_regression'
            
            
#         print("...Model set.")
       
    
    
#     def test_ranker(ranker_class, x, y, *args, **kwargs):
#         print('Initializing Ranker')
#         ranker = ranker_class(*args, **kwargs)
        
#         print('Ranking')
#         batch = ranker.rank(x,y)
#         print(batch)
        
#     print('testing default ranker')
#     test_ranker(Ranker,
#                 model.X_train[0:10],
#                 model.y_train[0:10],
#                 model.batch_size,
#                 how_many_batches=1,
#                 uncertainty_method='mc_dropout_5',
#                 diversity_method='no',
#                 highpred_method='no',
#                 uncertainty_weight=1,
#                 diversity_weight=0,
#                 highpred_weight=0,
#                 chosen_model=chosen_model,
#                 cycle=5,
#                 sigmadistr_freq=2,
#                 outdir='./,
#                 outflag='testing_ranker',
#                 device=device,
#                 local_seed=41, 
#                 task_type=task_type,
#                )
    
#     print('testing power ranker')
#     test_ranker(PowerRanker,
#                 model.X_train[0:10],
#                 model.y_train[0:10],
#                 model.batch_size,
#                 how_many_batches=1,
#                 beta=1
#                 uncertainty_method='mc_dropout_5',
#                 diversity_method='no',
#                 highpred_method='no',
#                 uncertainty_weight=1,
#                 diversity_weight=0,
#                 highpred_weight=0,
#                 chosen_model=chosen_model,
#                 cycle=5,
#                 sigmadistr_freq=2,
#                 outdir='./,
#                 outflag='testing_ranker',
#                 device=device,
#                 local_seed=41, 
#                 task_type=task_type,
#                )
    
#     print('testing softmax ranker')
#     test_ranker(SoftmaxRanker,
#                 model.X_train[0:10],
#                 model.y_train[0:10],
#                 model.batch_size,
#                 how_many_batches=1,
#                 beta=1
#                 uncertainty_method='mc_dropout_5',
#                 diversity_method='no',
#                 highpred_method='no',
#                 uncertainty_weight=1,
#                 diversity_weight=0,
#                 highpred_weight=0,
#                 chosen_model=chosen_model,
#                 cycle=5,
#                 sigmadistr_freq=2,
#                 outdir='./,
#                 outflag='testing_ranker',
#                 device=device,
#                 local_seed=41, 
#                 task_type=task_type,
#                )
    
#     print('testing softrank ranker')
#     test_ranker(SoftrankRanker,
#                 model.X_train[0:10],
#                 model.y_train[0:10],
#                 model.batch_size,
#                 how_many_batches=1,
#                 beta=1
#                 uncertainty_method='mc_dropout_5',
#                 diversity_method='no',
#                 highpred_method='no',
#                 uncertainty_weight=1,
#                 diversity_weight=0,
#                 highpred_weight=0,
#                 chosen_model=chosen_model,
#                 cycle=5,
#                 sigmadistr_freq=2,
#                 outdir='./,
#                 outflag='testing_ranker',
#                 device=device,
#                 local_seed=41, 
#                 task_type=task_type,
#                )
