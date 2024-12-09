import os
import numpy as np
import torch
import sys
sys.path.append('../')
import outflag_2_nickname
import tqdm
import h5py
import FUNCTIONS_4_DALdna
#from characterize_seqs import calculate_cross_sequence_identity_batch
import characterize_seqs
import matplotlib.pyplot as plt
import Experiments
import set_torch_tensors_test

def contained_method_1(X1,X2):
    #found=False
    found=0
    for i_x,x1 in tqdm.tqdm(enumerate(X1),total=len(X1)):
        #dna=FUNCTIONS_4_DALdna.ohe_to_seq(x,four_zeros_ok=False)
        #for j_x,x1 in tqdm.tqdm(enumerate(X_train),total=len(X_train)):
        for j_x,x2 in enumerate(X2):
            #dna1=FUNCTIONS_4_DALdna.ohe_to_seq(x1,four_zeros_ok=False)
            #print(dna)
            #print(dna1)
            #print()
            #if dna==dna1: 
            if (x1==x2).all(): 
                #found=True
                found+=1
                print("found")
                #break
        #if found: break
        #if x in X_train:
        #    found=True
    return found

def contained_method_2(pool,x_train_list):
    """Here are three versions -
    What I drew on the board: takes a list of pool seqs as strings and removes any matching a list containing X_train sequences as strings
    x_train_set = set(x_train_list)
    pool = [p for p in pool if not p in x_train_set]
    2. A better version that also deduplicates the list of pooled sequences
    x_train_set = set(x_train_list)
    pool = [p for p in set(pool) if not p in x_train_set]
    3. An even better version that does version 2 using the built in vectorization
    pool = list(set(pool)-set(x_train_list))
    """
    new_pool= list(set(pool)-set(x_train_list))
    return new_pool

def contained_method_3(pool,x_train):
    #new_pool= list(set(set_torch_tensors_test.lookup_ohe_to_dna(pool))-set(set_torch_tensors_test.lookup_ohe_to_dna(x_train_list)))
    new_pool=torch.tensor(np.array(set_torch_tensors_test.lookup_dna_to_ohe(list(set(set_torch_tensors_test.lookup_ohe_to_dna(pool))-set(set_torch_tensors_test.lookup_ohe_to_dna(x_train))))),dtype=torch.float32) 
    return new_pool

if __name__=='__main__':
    #args = parser.parse_args()
    #print(f"{args=}")
    
    wanted_temp=True
    #wanted_temp=False

    myfontsize=25 #18 #12
    myfigsize=(12,10) #plot_Res_AC.py: (10, 8)
    mydpi=600 #250
    myfigureformat='png' 

    #seqmeths=['totally_random']
    seqmeths=['mutation']
    #seqmethtestlist=['ID','mutation','evoaug','totally_random'] #,'simID']
    #nicks=['JRZtNzq1M']

    print("Getting nick dictionary...")
    #all_nicks=Experiments.get_nick_dictionary() #TEMP:DEFINITIVE
    print("Got it.")

    spacing=2

    #if args.exp==['all']:
    #    nicks=all_nicks
    #else:
        # nicks={}
        # for exp in args.exp:
        #     nicks[exp]=all_nicks[exp]

    #nicks=all_nicks  #TEMP:DEFINITIVE

    #seedoodloop=[51,52,53,54,55] #TEMP:DEFINITIVE
    seedoodloop=[51,52] #TEMP
    #seedoodloop=[51]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #i_al_loop=[1,4] #range(0,4+1)
    #i_al_loop=[1,2,3,4]
    #i_al_loop=args.i_al_loop #[0,1,2,3,4]
    #index_from_loop=[42] #range(41,45+1)
    #index_from_loop=[41,42,43,44,45] #TEMP:DEFINITIVE
    index_from_loop=[41,42] #TEMP
    i_al_loop=[4]
    #i_al=4
    #i_al_loop=['pristine','0','1','2','3','4']
    any_al=['pristine','0','1','2','3','4']

    #####

    if os.uname()[1]=='auros':
        mydir='../../outputs_DALdna_4plot/SAFECOPY_outputs_DALdna_4plot_22Apr2024/'
    elif os.uname()[1]=='citra':
        #mydir='../../imported_outputs_DALdna_4plot/'24
        mydir='../../outputs_DALdna_4plot/'
    elif os.uname()[1]=='citra':
        mydir='../../imported_outputs_DALdna_4plot/'
    elif os.uname()[1]=='comet':
        mydir='../../outputs_DALdna_4plot/'
        mydir1='../../imported_from_bamdev/'
        mydir2='../../outputs_DALdna/'
    else:
        mydir='../../outputs_DALdna_4plot/'
        mydir1='../../outputs_DALdna_4plot/'
        #mydir2='../../testfilesOOD/'
        mydir2='../../outputs_DALdna/'
    outpdir='../../outputs_DALdna_4plot/'

    #nickloop=nicks
    #for experim in nicks.keys(): #TEMP:DEFINITIVE
    for experim in ['temp']:  #TEMP
    #for experim in experims:
        print("=== === ===",experim)
        #nickloop=nicks[experim] #pre 11 Oct 2024
        if experim=='temp':
            nickloop=["JRZtNzt1O","JRZmNzv1O"] #,"JRZsNzv1O","JRZdNzv1O","JRZFNzv1O"]
        else:   
            nickloop=[Experiments.placeholder_nick(nick) for nick in nicks[experim]]

        fig = plt.figure(figsize=myfigsize) #20,13
        #for i_sq,seqmethtest in tqdm.tqdm(enumerate(seqmeths),total=len(seqmeths), desc='Loop over seqmeths',colour='red'):
        for i_sq,seqmethtest in enumerate(seqmeths):
            if seqmethtest=='mutation':
                #mutrateloop=[0.02, 0.05, 0.1, 0.2] #0.25
                mutrateloop=[0.05] #0.25
            elif seqmethtest=='evoaug':
                #mutrateloop=[0.02, 0.05, 0.1, 0.2] #0.25
                mutrateloop=[0.05] #0.25
            else:
                mutrateloop=[0.25] #dummy
            for mutrate in mutrateloop:
                #for nick in nickloop:
                #for i_n,nick in tqdm.tqdm(enumerate(nickloop),total=len(nickloop), desc='Loop over nicks',colour='blue'):
                for i_n,nick in enumerate(nickloop):
                    outflag=outflag_2_nickname.get_outflag_for_nick(nick).replace('mtrt-0.','mtrt-0p').replace('mtrt-1.','mtrt-1p')      
                    chosen_model=os.popen("python ../outflag_2_nickname.py --nick "+nick+" | tail -1 | awk '{print $2}'").read().replace('\n','')
                    chosen_dataset=os.popen("python ../outflag_2_nickname.py --nick "+nick+" | tail -1 | awk '{print $3}'").read().replace('\n','')  

                    if not os.path.isdir('../../outputs_DALdna_4plot/'+outflag):
                        #print("Directory not found:",'../../outputs_DALdna_4plot/'+outflag)
                        print("Directory not found for:",nick)
                    else:
                        percid_avg_allial=[]
                        percid_std_allial=[]
                        #for i_al in tqdm.tqdm(i_al_loop,desc='i_al_loop'):
                        for i_al in i_al_loop:

                            #testfile='../../outputs_DALdna/testfile__LegNetPK__newLentiMPRAK562_labels-seed0_random0_20000__mutation__0p02__seed-41.h5'
                            #dummy_h5f='../inputs/newLentiMPRAK562_labels-seed0_random0_20000.h5'
                            all_percid=[]
                            #all_percid=np.array([]) #pre 23 oct 2024
                            #for seedood in tqdm.tqdm(seedoodloop,desc='seedoodloop'):
                            for seedood in seedoodloop:
                                which_set='X_test'
                                #testfile='testfile__'+chosen_model+'__'+chosen_dataset+'__'+seqmethtest+'__'+str(mutrate).replace('.','p')+'__seed-'+str(seedood)+'.h5'
                                testfile='testfileO_'+which_set+'__'+chosen_model+'__'+chosen_dataset+'__'+seqmethtest+'__'+str(mutrate).replace('.','p')+'__seed-'+str(seedood)+'.h5'
                                if not os.path.isfile(mydir2+testfile):
                                    print("testfile not found: "+mydir2+testfile)
                                else:
                                    testdata=h5py.File(mydir2+testfile, 'r') 
                                    X_ood=torch.tensor(np.array(testdata['X_test']))

                                    #for index_from in tqdm.tqdm(index_from_loop,desc='index_from_loop'):
                                    for index_from in index_from_loop:
                                        #print(f"=== === === {nick=} {seqmethtest=} {mutrate=} {seedood=} {index_from=}")

                                        ### LASTPROPOSED
                                        if i_al=='pristine':
                                            input_h5_file=mydir+outflag+"/dal_dataset_pristine_seed-0-0.h5" 
                                        else:
                                            input_h5_file=mydir+outflag+"/dal_dataset_"+str(index_from)+"_proposed_iAL-"+str(i_al)+".h5" 
                                        #print(f"PERCIDDEBUG: {input_h5_file=} {index_from=} {i_al=} | {testfile=} {chosen_model=} {chosen_dataset=} {seqmethtest=} {mutrate=} {seedood=}")
                                        data = h5py.File(input_h5_file, 'r')
                                        X_train=torch.tensor(np.array(data['X_train']))

                                        ### CUMULATIVE
                                        # X_train=torch.empty(0)
                                        # for jal in range(6):
                                        #     ii_al=any_al[jal]
                                        #     if ii_al=='pristine':
                                        #         input_h5_file=mydir+outflag+"/dal_dataset_pristine_seed-0-0.h5" 
                                        #     else:
                                        #         input_h5_file=mydir+outflag+"/dal_dataset_"+str(index_from)+"_proposed_iAL-"+str(i_al)+".h5" 
                                        #     #print(f"PERCIDDEBUG: {input_h5_file=} {index_from=} {i_al=} | {testfile=} {chosen_model=} {chosen_dataset=} {seqmethtest=} {mutrate=} {seedood=}")
                                        #     data = h5py.File(input_h5_file, 'r')
                                        #     X_train_iial=torch.tensor(np.array(data['X_train']))
                                        #     X_train=torch.cat((X_train,X_train_iial),axis=0)
                                        #     if ii_al==i_al: break

                                        print(f"{X_ood.shape=} {X_train.shape=}")

                                        """
                                        found=contained_method_1(X_ood,X_train)
                                        print(f"{seedood=} {index_from=} {found=}")
                                        """
                                        
                                        new_X_ood=contained_method_3(X_ood,X_train) 
                                        print(f"{len(new_X_ood)=} {len(X_ood)=}")
                                        #div_ood=characterize_seqs.sequence_diversity(X_ood,batch_size=256)
                                        #div_train=characterize_seqs.sequence_diversity(X_train,batch_size=256)
                                        #div_ood=characterize_seqs.sequence_diversity_onerow(X_ood,batch_size=256)
                                        #div_train=characterize_seqs.sequence_diversity_onerow(X_train,batch_size=256)
                                        #print(f"{div_ood=} {div_train=}")

                                        #percent_identity = calculate_cross_sequence_identity_batch(X_train, X_ood, batch_size=100) # pre 23 oct 2024
                                        #np.save(outpdir+'percidentity_'+whichseqs+'_'+nick+'_seedadd-'+str(index_from)+'_iAL-'+str(i_al)+'.npy',np.array(percent_identity)) 
                                        #percid_avg=np.mean(percent_identity.flatten())
                                        #percid_std=np.std(percent_identity.flatten())
                                        #print(f"{percid_avg=} {percid_std=}")
                                        #print(f"Useful to get a sense of the values: {np.mean(percent_identity.flatten())=}") #pre 23 oct 2024
                                        #print("pre concat")
                                        #for percid in percent_identity.flatten():
                                        #    all_percid.append(percid)
                                        #all_percid=np.concatenate((all_percid,percent_identity.flatten())) # pre 23 oct 2024
                                        #print("post concat")
                                        # np.save(outpdir+'percidentitymean_'+whichseqs+'_'+nick+'_seedadd-'+str(index_from)+'_iAL-'+str(i_al)+'.npy',np.mean(percent_identity.flatten())) 
                                        # np.save(outpdir+'percidentitystd_'+whichseqs+'_'+nick+'_seedadd-'+str(index_from)+'_iAL-'+str(i_al)+'.npy',np.std(percent_identity.flatten())) 

                                        max_percent_identity_1,average_max_percent_identity_1,global_max_percent_identity_1, mean_percent_identity_1, mean_axis1, std_axis1 = characterize_seqs.percid_functions(X_train, X_ood, batch_size=256)
                                        #print(f"Useful to get a sense of the values: {nick=} {seqmethtest=} {index_from=} {seedood=} {average_max_percent_identity_1=} {mean_axis1=} {std_axis1=}") # {div_train=} {div_ood=}")
                                        print(f"Useful to get a sense of the values: {nick=} {seqmethtest=} {index_from=} {seedood=} {average_max_percent_identity_1=} {mean_percent_identity_1=}") # {div_train=} {div_ood=}")
                                        # max_percent_identity_1,average_max_percent_identity_1,global_max_percent_identity_1, mean_percent_identity_1, mean_axis1, std_axis1 = characterize_seqs.percid_functions(X_train, X_train, batch_size=256)
                                        # print(f"Useful to get a sense of the values (train-train): {nick=} {seqmethtest=} {index_from=} {seedood=} {average_max_percent_identity_1=} {mean_axis1=} {std_axis1=}")
                                        # max_percent_identity_1,average_max_percent_identity_1,global_max_percent_identity_1, mean_percent_identity_1, mean_axis1, std_axis1 = characterize_seqs.percid_functions(X_ood, X_ood, batch_size=256)
                                        # print(f"Useful to get a sense of the values (ood-ood): {nick=} {seqmethtest=} {index_from=} {seedood=} {average_max_percent_identity_1=} {mean_axis1=} {std_axis1=}")
                                        
                                        #all_percid=np.concatenate((all_percid,average_max_percent_identity_1))
                                        all_percid.append(average_max_percent_identity_1)
                            print(f"{nick=} {seqmethtest=} {i_al=} {np.mean(all_percid)=} {np.std(all_percid)=}")
                            percid_avg_allial.append(np.mean(all_percid))
                            percid_std_allial.append(np.std(all_percid))

                        curr_idx=i_n+i_sq*(len(nickloop)+spacing) #grouped by seqmethtest
                        positions=np.arange(len(percid_avg_allial))+curr_idx
                        short_dict=outflag_2_nickname.make_short_dict(nick) #,request=experim)
                        plt.bar(positions,percid_avg_allial,color=short_dict[nick]['color'], lw=2, label=short_dict[nick]['plotname'])  
                        plt.errorbar(positions,percid_avg_allial,yerr=percid_std_allial,color='black')

            # plt.plot(xaxis,npoints_avg,color=mycolor, lw=1, label=nick,alpha=1.0) 
            # plt.fill_between(xaxis, npoints_avg-npoints_std, npoints_avg+npoints_std, alpha=0.5, color=mycolor)
            # plt.scatter(xaxis,npoints_avg,color=mycolor, lw=1,alpha=1.0) 

            plt.legend(loc='lower right',prop={'size':15})
            plt.xlabel('Cycles')
            plt.ylabel('Avg. Perc. Id.')
            outfile='../../outputs_DALdna_4plot/Char_OODPercid_'+experim+'.png'
            fig.savefig(outfile,dpi=mydpi, bbox_inches='tight') 
            print("DONE:",outfile)

    print("SCRIPT END.")

"""
To monitor:
grep === nohup.is_.out  

"""