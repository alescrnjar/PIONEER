import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--outflag',default='Model-ResidualBind_DS-VTS1_rnacompete2009labels-seed0_random0_5000_pN-5000_gU-5000_mxe-300_ALc-1_itr-retrain_bao-50_seqmeth-mutation_pristmeth-ds_unc-no-0p0_div-no-0p0_hpred-no-0p0_spdes-mc_dropout_5')
parser.add_argument('--nick',default='VR5mtN5x',type=str)
parser.add_argument('--old',default='0',type=str)


def get_nicks():
    dsnick={
    #'V':'VTS1_rnacompete2009',
    'T':'VTS1_rnacompete2013',
    'X':'RBFOX1_rnacompete2013',
    'D':'DeepSTARRdev_',
    'L':'LentiMPRA_',
    'P':'HepG2tsv_',
    'G':'LentiMPRA_',
    'm':'LentiMPRA_',
    'K':'newLentiMPRAK562_',
    'a':'newLentiMPRAK562two_',
    'H':'newLentiMPRAHepG2_',
    'b':'newLentiMPRAHepG2two_',
    'J':'newLentiMPRAK562_',
    'c':'newLentiMPRAK562two_',
    'Q':'newLentiMPRAHepG2_',
    'd':'newLentiMPRAHepG2two_',
    }
    seqnick={
'mt':'mutation',   #m
'sa':'simulated_annealing', #a
'sl':'saliency', #s
'sly':'saliency_y', #y
'gr':'greedy', #r
'ga':'genetic', #g
'tr':'totally_random', #t
'ev':'evoaug', #e
'hs':'hessian', #h
'say':'simulated_annealing_y', #z
'gay':'genetic_y', #q
'pl':'XdsYor', #p
'sv':'saliency_div_y', #v
    }
    seqnicknew={
'm':'mutation',   #m
'a':'simulated_annealing', #a
's':'saliency', #s
'y':'saliency_y', #y
'r':'greedy', #r
'g':'genetic', #g
't':'totally_random', #t
'e':'evoaug', #e
'E':'evoaugassign', #E
'_':'evoaugmut', #E
'h':'hessian', #h
#'z':'simulated_annealing_y', #z
'q':'genetic_y', #q
'p':'XdsYor', #p
'v':'saliency_div_y', #v
'n':'saliency_U-A',
'W':'totally_random_then_saliency', 
'U':'dinuc_then_saliency',
'Y':'mutation_then_saliency',
'F':'fromfile',
'H':'saliency_div_hamming',
'K':'fromfile_then_saliency',
'd':'dinuc',
'z':'vanilla_diffusion',
'x':'diffusion_file',
'0':'diffusion_y',
'c':'saliency_aleat',
'i':'saliency_evidential',
'G':'GradientSHAP',
'L':'DeepLiftSHAP',
'b':'motifembed',
'B':'dinuc_then_motifembed',
'o':'realmut',
'1':'BatchBALDsubsel',
'3':'BADGEsubsel',
'4':'LCMDsubsel',
'9':'BADGEfromt',
'-':'concatBADGE1',
'8':'concatrand1',
'7':'concatLCMD1',
'6':'LCMDfromt',
'5':'LCMDfromd',
'Q':'LCMDfromJ',
'j':'realsal',
':':'realevoaug',
'C':'Costmixrand1',
#'L':'CostmixLCMD1', # This method is not a thing

'J':'realTEMPAsal',
'A':'concatTEMPArand1',
'D':'concatTEMPALCMD1',
'X':'CostmixTEMPArand1',
#'l':'CostmixTEMPALCMD1', #This method is not a thing
'P':'PriceHundredLCMD',

'k':'Price20KLCMD',
    }

    subsnick={
'N':'no-0.0-no-0.0-no-0.0',
'M':'mc_dropout_5-1.0-no-0.0-no-0.0',
'D':'sigma_deep_ensemble-1.0-no-0.0-no-0.0',
'E':'EL2N-1.0-no-0.0-no-0.0',
'N5x':'no-0.0-no-0.0-no-0.0',
'M5x':'mc_dropout_5-1.0-no-0.0-no-0.0',
    }
    dstrnick={
        'R':'random0',
        '2':'random2',
        '3':'random3',
        '4':'random4',
        '5':'random5',

        '8':'q085',
        '1':'negq015',
        'U':'uniform',
        'O':'original',
        'L':'leastac',
        'P':'fromzero',
        'M':'mostac',
        'N':'neutral',
        'T':'totrand0'
    }
    numbernick={
'5':'5000',
'O':'1000',
'T':'10000',
'Z':'20000', #zwanzig
'Q':'25000',
'D':'30000', #dreizig #
'L':'50000',
'S':'75000',
'C':'100000',
'P':'130000',
'B':'150000',

'j':'6400',
'W':'32000',
'J':'64000',
'N':'96000',
'4':'4000',
    }
    converted={
    'no-0.0-no-0.0-no-0.0':'unc-no-0p0_div-no-0p0_hpred-no-0p0',
    'mc_dropout_5-1.0-no-0.0-no-0.0':'unc-mc_dropout_5-1p0_div-no-0p0_hpred-no-0p0',
    'sigma_deep_ensemble-1.0-no-0.0-no-0.0':'unc-sigma_deep_ensemble-1p0_div-no-0p0_hpred-no-0p0',
    'EL2N-1.0-no-0.0-no-0.0':'unc-EL2N-1p0_div-no-0p0_hpred-no-0p0'
    }
    mtrtnick={
        'z':'0.0', #just as a test
        'g':'0.01', #genome variability
        'o':'0.03', #one mutation # int(39*0.026)=1,int(39*0.025)=0,int(41*0.025)=1,int(41*0.024)=0  0.026 is the minimum for 39, for 41 is 0.025. Picking 0.03 is the same of 0.026 since it also result in 1mutation
        'v':'0.05',
        't':'0.1', #tenth
        'q':'0.25', #quarter
        'h':'0.5', #half
        's':'0.75', #seven
        'f':'1.0', #full
    }
    baonick={
        ##'':'l', #bao 50: 50*100:5000 to be selected
        ##'5x':'x', #bao 10: 10*100:1000 to be selected
        'x':'10', #bao 10: 10*100:1000 to be selected
        'l':'50', #bao 50: 50*100:5000 to be selected #bao
        'c':'100', #bao 100: 100*100:10,000 to be selected #bao
        'q':'250', #quarter bao 250: 250*100:25,000 to be selected        #128*250=32K to be selected
        'z':'200', #quarter bao 200: 200*100:20,000 to be selected        
        'j':'500', #bao 500: 500*100:50,000 to be selected               #128*500=64K to be selected
        'D':'300',
        's':'750',                                                      #128*750=96K to be selected

        'M':'1000',
        'U':'1200',
        'V':'1500',
        'O':'1800',    
        '4':'40'
    }
    n2mnick={
        '1':'same',
        '5':'five',
        '6':'six',
        'T':'ten',
        'X':'twentyfaiv',
        'i':'na_1',
        'v':'na_5',
        'a':'scr1',
        'e':'scr5',

        'f':'fifth',
        'p':'thirteenth',
        's':'sixth',
        'h':'half',
        'D':'dreizig',
    }
    spdesnick={
        'M':'mc_dropout_5',
        'D':'sigma_deep_ensemble',
        'O':'mc_dropout_50deepens',
        'S':'sigma_deep_ensemble0deepens',
        'E':'EL2N'
    }

    return dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick

def get_old_outflags_for_nick(nick): # VR5slN VR5slN5x
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()

    what_dataset=dsnick[nick[0]]
    init_distrib=dstrnick[nick[1]]
    init_number=numbernick[nick[2]]
    subsel=subsnick[nick[5:]]
    seq_method=seqnick[nick[3:5]]

    genU='5000'
    if init_number!='5000': #B2
        genU=init_number

    hmnbao='50'
    if '5x' in nick[5:]: #A2
        hmnbao='10'

    outflag_old='Model-ResidualBind_DS-'+what_dataset+'labels-seed0_'+init_distrib+'_'+init_number
    outflag_old+='_pN-'+init_number #QUIQUIURG correct?
    outflag_old+='_gU-'+genU
    outflag_old+='_mxe-300'
    outflag_old+='_ALc-1'
    outflag_old+='_itr-retrain'
    outflag_old+='_bao-'+hmnbao
    outflag_old+='_seqmeth-'+seq_method
    outflag_old+='_pristmeth-ds'
    outflag_old+='_'+converted[subsel]
    outflag_old+='_spdes-mc_dropout_5'
    return outflag_old

def get_args_for_nick(nick): # VR5sNlq1M
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()

    what_dataset=dsnick[nick[0]] #V
    init_distrib=dstrnick[nick[1]] #R
    init_number=numbernick[nick[2]] #5
    subsel=subsnick[nick[4]] #N
    seq_method=seqnicknew[nick[3]] #s
    bao=baonick[nick[5]] #l
    mtrate=mtrtnick[nick[6]] #q
    n2make=n2mnick[nick[7]] #1

    #spdes='mc_dropout_5'
    spdes=spdesnick[nick[8]] #M #4deepens

    #genU=int(init_number) #'5000'
    #genU='5000' #pre 31 mar 2024
    #if init_number!='5000': #B2
    #    genU=init_number
    #if n2make!='same': ## or n2make=='na_1': #goodold #pre 31 mar 2024
    #    genU=str(int(init_number)*int(nick[7])) 
    if n2make=='same': ## or n2make=='na_1': 
        genU=init_number 
    elif n2make=='na_1':
        genU='notanch-'+init_number
    elif n2make=='five': ## or n2make=='na_1': 
        genU=str(int(init_number)*5) #int(nick[7])) 
    elif n2make=='ten': ## or n2make=='na_1': 
        genU=str(int(init_number)*10) #int(nick[7])) 
    elif n2make=='twentyfaiv': ## or n2make=='na_1': 
        genU=str(int(init_number)*25) #int(nick[7])) 
    elif n2make=='six': ## or n2make=='na_1': 
        genU=str(int(init_number)*6) #int(nick[7])) 
    elif n2make=='na_5':
        genU='notanch-'+str(int(init_number)*5)
    elif n2make=='scr1':
        genU='screm-'+str(int(init_number)*1)
    elif n2make=='scr5':
        genU='screm-'+str(int(init_number)*5)
    elif n2make=='fifth':
        genU=str(int(int(init_number)/5))
    elif n2make=='sixth':
        genU=str(int(int(init_number)/6))
    elif n2make=='dreizig':
        genU=str(int(int(init_number)/30))
    elif n2make=='thirteenth':
        genU=str(int(int(init_number)/13))
    elif n2make=='half':
        genU=str(int(int(init_number)/2))
    else:
        print("ERROR: wrong n2make")
        exit()

    if nick[0]=='V':
        chosen_model='ResidualBind'
    elif nick[0]=='T':
        chosen_model='ResidualBind'
    elif nick[0]=='X':
        chosen_model='ResidualBind'
    elif nick[0]=='L':
        chosen_model='ResidualBind'
    elif nick[0]=='G':
        chosen_model='LegNet'
    elif nick[0]=='P':
        chosen_model='LegNet_Custom'
    elif nick[0]=='m':
        chosen_model='mpra'
    elif nick[0]=='D':
        chosen_model='DeepSTARR'
    elif nick[0]=='K':
        chosen_model='NewResNet'
    elif nick[0]=='a':
        chosen_model='NewResNet'
    elif nick[0]=='H':
        chosen_model='NewResNet'
    elif nick[0]=='b':
        chosen_model='NewResNet'
    elif nick[0]=='J':
        chosen_model='LegNet_Custom'
    elif nick[0]=='c':
        chosen_model='LegNet_Custom'
    elif nick[0]=='Q':
        chosen_model='LegNet_Custom'
    elif nick[0]=='d':
        chosen_model='LegNet_Custom'
    else:
        print("ERROR: wrong dataset choice:",nick[0])
        exit()

    if chosen_model=='ResidualBind':
        trainmaxep=300
    elif chosen_model=='NewResNet':
        trainmaxep=300
    elif chosen_model=='DeepSTARR':
        trainmaxep=100
    elif chosen_model=='LegNet':
        trainmaxep=100
    elif chosen_model=='LegNet_Custom':
        trainmaxep=100
    elif chosen_model=='mpra':
        trainmaxep=100
    else:
        print("ERROR: wrong chosen_model")
        exit()

    return chosen_model, what_dataset, init_distrib, init_number, genU, trainmaxep, bao, seq_method, mtrate,subsel,spdes

def make_short_dict(nick,request=''):
    mycolor='black' #AC: default
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()
    chosen_model, what_dataset, init_distrib, init_number, genU, trainmaxep, bao, seq_method, mtrate,subsel,spdes=get_args_for_nick(nick)

    chosen_dataset=what_dataset+'labels-seed0_'+init_distrib+'_'+init_number
    #print(f"make_short_dict: {what_dataset=} {mtrate=} {subsel=}")

    plottitle=nick #'NOT FOUND'
    mycolor='black'
    hatch=''
    #hatch_patterns = ['/', '\\', '|', '-', '+']  # Define hatch patterns                                                                                                                                                                                  

                                              # Cost
    if nick in ['JRZmNzv1O','JRZmNzviO',      'JRZmNMv5O','JRZmNMvvO',       'JRZmNzq1O','JRZmNzqiO',      'JRZmNMq5O','JRZmNMqvO',
                'JTZmNzv1O','JTZmNzviO',      'JTZmNMv5O','JTZmNMvvO',
                'QRZmNzv1O','QRZmNzviO',      'QRZmNMv5O','QRZmNMvvO',       'QRZmNzq1O','QRZmNzqiO',      'QRZmNMq5O','QRZmNMqvO',
                'QTZmNzv1O','QTZmNzviO',      'QTZmNMv5O','QTZmNMvvO',
                'XRZmNzv1O','XRZmNzviO',      'XRZmNMv5O','XRZmNMvvO',       'XRZmNzq1O','XRZmNzqiO',      'XRZmNMq5O','XRZmNMqvO',
                'XTZmNzv1O','XTZmNzviO',      'XTZmNMv5O','XTZmNMvvO',
                
                'DRZmNzq1O'
                ]:
        plottitle='Mutagenesis (multicycle)'
        mycolor='darkgoldenrod'
    if nick in ['JRZoNzv1O','JRZoNzviO',      'JRZoNMv5O','JRZoNMvvO',       'JRZoNzq1O','JRZoNzqiO',      'JRZoNMq5O','JRZoNMqvO',
                'JTZoNzv1O','JTZoNzviO',      'JTZoNMv5O','JTZoNMvvO',
                'QRZoNzv1O','QRZoNzviO',      'QRZoNMv5O','QRZoNMvvO',       'QRZoNzq1O','QRZoNzqiO',      'QRZoNMq5O','QRZoNMqvO',
                'QTZoNzv1O','QTZoNzviO',      'QTZoNMv5O','QTZoNMvvO',
                'XRZoNzv1O','XRZoNzviO',      'XRZoNMv5O','XRZoNMvvO',       'XRZoNzq1O','XRZoNzqiO',      'XRZoNMq5O','XRZoNMqvO',
                'XTZoNzv1O','XTZoNzviO',      'XTZoNMv5O','XTZoNMvvO',

                'DRZoNzv1O','DRZoNzviO',      'DRZoNMv5O','DRZoNMvvO',       'DRZoNzq1O','DRZoNzqiO',      'DRZoNMq5O','DRZoNMqvO',
                'DTZoNzv1O','DTZoNzviO',      'DTZoNMv5O','DTZoNMvvO',
                ]:
        plottitle='Mutagenesis' # (1 cycle)'
        mycolor='#ffae00ff'
    if nick in ['JRZtNzq1O','JRZtNzqiO','JRZtNzt1O','JRZtNztiO','JRZtNzv1O','JRZtNzviO',      'JRZtNMq5O','JRZtNMqvO','JRZtNMt5O','JRZtNMtvO','JRZtNMv5O','JRZtNMvvO',
                'JTZtNzq1O','JTZtNzqiO','JTZtNzt1O','JTZtNztiO','JTZtNzv1O','JTZtNzviO',      'JTZtNMq5O','JTZtNMqvO','JTZtNMt5O','JTZtNMtvO','JTZtNMv5O','JTZtNMvvO',
                
                'QRZtNzq1O','QRZtNzqiO','QRZtNzt1O','QRZtNztiO','QRZtNzv1O','QRZtNzviO',      'QRZtNMq5O','QRZtNMqvO','QRZtNMt5O','QRZtNMtvO','QRZtNMv5O','QRZtNMvvO',
                'QTZtNzq1O','QTZtNzqiO','QTZtNzt1O','QTZtNztiO','QTZtNzv1O','QTZtNzviO',      'QTZtNMq5O','QTZtNMqvO','QTZtNMt5O','QTZtNMtvO','QTZtNMv5O','QTZtNMvvO',
                
                'XRZtNzq1O','XRZtNzqiO','XRZtNzt1O','XRZtNztiO','XRZtNzv1O','XRZtNzviO',      'XRZtNMq5O','XRZtNMqvO','XRZtNMt5O','XRZtNMtvO','XRZtNMv5O','XRZtNMvvO',
                'XTZtNzq1O','XTZtNzqiO','XTZtNzt1O','XTZtNztiO','XTZtNzv1O','XTZtNzviO',      'XTZtNMq5O','XTZtNMqvO','XTZtNMt5O','XTZtNMtvO','XTZtNMv5O','XTZtNMvvO',
        
                'DRZtNzq1O','DRZtNzqiO','DRZtNzt1O','DRZtNztiO','DRZtNzv1O','DRZtNzviO',      'DRZtNMq5O','DRZtNMqvO','DRZtNMt5O','DRZtNMtvO','DRZtNMv5O','DRZtNMvvO',
                ]:
        plottitle='Random'
        mycolor='C3' 
    if nick in ['JRZdNzv1O', 'QRZdNzv1O', 'XRZdNzv1O',
                'JTZdNzv1O', 'QTZdNzv1O', 
                ]:
        plottitle='Dinuc. shuffle'
        mycolor='darkred' 

    if nick in ['JRZsNzv1O','JRZsNzviO',     'JRZsNzq1O','JRZsNzqiO',
                'JTZsNzv1O','JTZsNzviO',
                'QRZsNzv1O','QRZsNzviO',     'QRZsNzq1O','QRZsNzqiO',
                'QTZsNzv1O','QTZsNzviO',
                'XRZsNzv1O','XRZsNzviO',     'XRZsNzq1O','XRZsNzqiO',
                'XTZsNzv1O','XTZsNzviO',

                'DRZsNzq1O',
                ]:
        plottitle='UGM (multicycle)' #'Unc. Backp.'
        mycolor='blue'

    if nick in [
                #'JRZjNzv1O','JRZjNzviO',     'JRZjNzq1O','JRZjNzqiO',
                #'JTZjNzv1O','JTZjNzviO',
                #'QRZjNzv1O','QRZjNzviO',     'QRZjNzq1O','QRZjNzqiO',
                #'QTZjNzv1O','QTZjNzviO',
                #'XRZjNzv1O','XRZjNzviO',     'XRZjNzq1O','XRZjNzqiO',
                #'XTZjNzv1O','XTZjNzviO',

                #'DRZjNzq1O',

                'JRZJNzv1O','JRZJNzviO',     'JRZJNzq1O','JRZJNzqiO',
                'JTZJNzv1O','JTZJNzviO',
                'QRZJNzv1O','QRZJNzviO',     'QRZJNzq1O','QRZJNzqiO',
                'QTZJNzv1O','QTZJNzviO',
                'XRZJNzv1O','XRZJNzviO',     'XRZJNzq1O','XRZJNzqiO',
                'XTZJNzv1O','XTZJNzviO',

                'DRZJNzv1O','DRZJNzviO',     'DRZJNzq1O','DRZJNzqiO',
                'DTZJNzv1O','DTZJNzviO',
                ]:
        plottitle='UGM' # (1 cycle)' #'Unc. Backp.'
        mycolor='C0'
    if nick in [
                'JRZjNzv1O','JRZjNzviO',     'JRZjNzq1O','JRZjNzqiO',
                'JTZjNzv1O','JTZjNzviO',
                'QRZjNzv1O','QRZjNzviO',     'QRZjNzq1O','QRZjNzqiO',
                'QTZjNzv1O','QTZjNzviO',
                'XRZjNzv1O','XRZjNzviO',     'XRZjNzq1O','XRZjNzqiO',
                'XTZjNzv1O','XTZjNzviO',

                'DRZjNzq1O',
                ]:
        plottitle='UGM (stochastic)' # (1 cycle)' #'Unc. Backp.'
        mycolor='C0'

    if nick in ['JRZFNzq1O','JRZFNzv1O',
                'QRZFNzq1O','QRZFNzv1O',
                'DRZFNzq1O','DRZFNzv1O','DRZFNzviO',
                ]:
        plottitle='Genome' #'Unc. Backp.'
        if 'RBFOX1' in chosen_dataset: plottitle='RNAcompete'
        mycolor='grey'

    if nick in ['JRZyNzv1O','JRZyNzviO',]:
        plottitle='Activity' 
        mycolor='turquoise'
    if nick in ['JRZvNzv1O','JRZvNzviO']:
        plottitle='Activity Diversity' 
        mycolor='aquamarine'

    if nick in ['JRZtMzv5O','JRZtMzvvO',     'JRZtMzq5O','JRZtMzqvO',
                'QRZtMzv5O','QRZtMzvvO',     'QRZtMzq5O','QRZtMzqvO',
                'DRZtMzv5O','DRZtMzvvO',     'DRZtMzq5O','DRZtMzqvO',
                ]:
        plottitle='Random + Unc.'
        #mycolor='forestgreen'
        #mycolor='lightcoral'
        mycolor='C3'
        hatch='///'
    # if nick in ['JRZmMzv5O','JRZmMzvvO',     'JRZmMzq5O','JRZmMzqvO',
    #             'JTZmMzv5O','JTZmMzvvO',
    #             'QRZmMzv5O','QRZmMzvvO',     'QRZmMzq5O','QRZmMzqvO',
    #             'QTZmMzv5O','QTZmMzvvO',
    #             'XRZmMzv5O','XRZmMzvvO',     'XRZmMzq5O','XRZmMzqvO',
    #             'XTZmMzv5O','XTZmMzvvO',
    #             ]:
    if nick in ['JRZoMzv5O','JRZoMzvvO',     'JRZoMzq5O','JRZoMzqvO',
                'QRZoMzv5O','QRZoMzvvO',     'QRZoMzq5O','QRZoMzqvO',
                'DRZoMzv5O','DRZoMzvvO',     'DRZoMzq5O','DRZoMzqvO',
                
                ]:
        plottitle='Mutagenesis + Unc.'
        #mycolor='yellowgreen'
        #mycolor='coral'
        mycolor='#ffae00ff'
        #mycolor='lightsteelblue'
        hatch='///'
    if nick in ['JRZdMzv5O','JRZdMzvvO',     'JRZdMzq5O','JRZdMzqvO',
                'JTZdMzv5O','JTZdMzvvO',
                'QRZdMzv5O','QRZdMzvvO',     'QRZdMzq5O','QRZdMzqvO',
                'QTZdMzv5O','QTZdMzvvO',
                'XRZdMzv5O','XRZdMzvvO',     'XRZdMzq5O','XRZdMzqvO',
                'XTZdMzv5O','XTZdMzvvO',
                ]:
        plottitle='Dinuc.Sh. + Unc. Subsel.'
        mycolor='tan'
    if nick in ['JRZJMzv5O','JRZJMzvvO',     'JRZJMzq5O','JRZJMzqvO',
                'JTZJMzv5O','JTZJMzvvO',
                'QRZJMzv5O','QRZJMzvvO',     'QRZJMzq5O','QRZJMzqvO',
                'QTZJMzv5O','QTZJMzvvO',
                'XRZJMzv5O','XRZJMzvvO',     'XRZJMzq5O','XRZJMzqvO',
                'XTZJMzv5O','XTZJMzvvO',
                ]:
        plottitle='UGM + Unc.'
        #mycolor='lightsteelblue'
        mycolor='C0'
        hatch='///'

    if nick in ['JRZ-Nzv1O','JRZ-NzviO']:
        plottitle='x3 pool + BADGE'
        mycolor='salmon'  

    # Cost
    is_cost='no'
    if nick in [
                'JRZmNMq5O','JRZmNMqvO', 'JRZtNMq5O','JRZtNMqvO',   'JRZmNMt5O','JRZmNMtvO', 'JRZtNMt5O','JRZtNMtvO',   'JRZmNMv5O','JRZmNMvvO', 'JRZtNMv5O','JRZtNMvvO',
                'JTZmNMq5O','JTZmNMqvO', 'JTZtNMq5O','JTZtNMqvO',   'JTZmNMt5O','JTZmNMvvO', 'JTZtNMt5O','JTZtNMtvO',   'JTZmNMv5O','JTZmNMvvO', 'JTZtNMv5O','JTZtNMvvO',
                'QRZmNMq5O','QRZmNMqvO', 'QRZtNMq5O','QRZtNMqvO',   'QRZmNMt5O','QRZmNMtvO', 'QRZtNMt5O','QRZtNMtvO',   'QRZmNMv5O','QRZmNMvvO', 'QRZtNMv5O','QRZtNMvvO',
                'QTZmNMq5O','QTZmNMqvO', 'QTZtNMq5O','QTZtNMqvO',   'QTZmNMt5O','QTZmNMvvO', 'QTZtNMt5O','QTZtNMtvO',   'QTZmNMv5O','QTZmNMvvO', 'QTZtNMv5O','QTZtNMvvO',
                'XRZmNMq5O','XRZmNMqvO', 'XRZtNMq5O','XRZtNMqvO',   'XRZmNMt5O','XRZmNMtvO', 'XRZtNMt5O','XRZtNMtvO',   'XRZmNMv5O','XRZmNMvvO', 'XRZtNMv5O','XRZtNMvvO',
                'XTZmNMq5O','XTZmNMqvO', 'XTZtNMq5O','XTZtNMqvO',   'XTZmNMt5O','XTZmNMvvO', 'XTZtNMt5O','XTZtNMtvO',   'XTZmNMv5O','XTZmNMvvO', 'XTZtNMv5O','XTZtNMvvO',
                
                'JRZoNMq5O','JRZoNMqvO', 'JRZdNMq5O','JRZdNMqvO',   'JRZoNMt5O','JRZoNMtvO', 'JRZdNMt5O','JRZdNMtvO',   'JRZoNMv5O','JRZoNMvvO', 'JRZdNMv5O','JRZdNMvvO',
                'JTZoNMq5O','JTZoNMqvO', 'JTZdNMq5O','JTZdNMqvO',   'JTZoNMt5O','JTZoNMvvO', 'JTZdNMt5O','JTZdNMtvO',   'JTZoNMv5O','JTZoNMvvO', 'JTZdNMv5O','JTZdNMvvO',
                'QRZoNMq5O','QRZoNMqvO', 'QRZdNMq5O','QRZdNMqvO',   'QRZoNMt5O','QRZoNMtvO', 'QRZdNMt5O','QRZdNMtvO',   'QRZoNMv5O','QRZoNMvvO', 'QRZdNMv5O','QRZdNMvvO',
                'QTZoNMq5O','QTZoNMqvO', 'QTZdNMq5O','QTZdNMqvO',   'QTZoNMt5O','QTZoNMvvO', 'QTZdNMt5O','QTZdNMtvO',   'QTZoNMv5O','QTZoNMvvO', 'QTZdNMv5O','QTZdNMvvO',
                'XRZoNMq5O','XRZoNMqvO', 'XRZdNMq5O','XRZdNMqvO',   'XRZoNMt5O','XRZoNMtvO', 'XRZdNMt5O','XRZdNMtvO',   'XRZoNMv5O','XRZoNMvvO', 'XRZdNMv5O','XRZdNMvvO',
                'XTZoNMq5O','XTZoNMqvO', 'XTZdNMq5O','XTZdNMqvO',   'XTZoNMt5O','XTZoNMvvO', 'XTZdNMt5O','XTZdNMtvO',   'XTZoNMv5O','XTZoNMvvO', 'XTZdNMv5O','XTZdNMvvO',
                ]:
        plottitle+=' (+100K)'
        is_cost='yes'

    if nick in ['JRZJNzv1S','JRZJNzviS','QRZJNzv1S','QRZJNzviS','DRZJNzv1S','DRZJNzviS']:
        seqtitle='UGM (Deep Ensemble)'
        plottitle='UGM (Deep Ensemble)'
        mycolor='C0'
        hatch='|'

    if nick in ['JRZ1Nzv5O','JRZ1NzvvO',    'JRZ1Nzq5O','JRZ1NzqvO',
                'JTZ1Nzv5O','JTZ1NzvvO',
                'QRZ1Nzv5O','QRZ1NzvvO',    'QRZ1Nzq5O','QRZ1NzqvO',
                'QTZ1Nzv5O','QTZ1NzvvO',
                'XRZ1Nzv5O','XRZ1NzvvO',    'XRZ1Nzq5O','XRZ1NzqvO',
                'XTZ1Nzv5O','XTZ1NzvvO',
                ]:
        plottitle='MaxDet (BatchBALD)'
        mycolor='yellowgreen'
    if nick in ['JRZ3Nzv5O','JRZ3NzvvO',    'JRZ3Nzq5O','JRZ3NzqvO',
                'JTZ3Nzv5O','JTZ3NzvvO',
                'QRZ3Nzv5O','QRZ3NzvvO',    'QRZ3Nzq5O','QRZ3NzqvO',
                'QTZ3Nzv5O','QTZ3NzvvO',
                'XRZ3Nzv5O','XRZ3NzvvO',    'XRZ3Nzq5O','XRZ3NzqvO',
                'XTZ3Nzv5O','XTZ3NzvvO',
                ]: 
        plottitle='KMeansPP (BADGE)'
        #mycolor='blueviolet'
        mycolor='C2'
    if nick in ['JRZ4Nzv5O','QRZ4Nzv5O','DRZ4Nzv5O',
                'JRZ4NzvvO','QRZ4NzvvO','DRZ4NzvvO',
                ]:
        plottitle='Mutagenesis + Batch'
        #mycolor='mediumseagreen'
        #mycolor='darkgoldenrod'
        mycolor='#ffae00ff'
        hatch='\\\\'
    if nick in ['JRZ5Nzv5O','QRZ5Nzv5O','DRZ5Nzv5O',
                'JRZ5NzvvO','QRZ5NzvvO','DRZ5NzvvO',
                ]:
        plottitle='dinuc. + Batch'
        mycolor='mediumseagreen'
    if nick in ['JRZ6Nzv5O','QRZ6Nzv5O','DRZ6Nzv5O',
                'JRZ6NzvvO','QRZ6NzvvO','DRZ6NzvvO',
                ]:
        plottitle='Random + Batch'
        #mycolor='mediumseagreen'
        #mycolor='crimson'
        #mycolor='darkred'
        mycolor='C3'
        hatch='\\\\'
    if nick in ['JRZQNzv5O','QRZQNzv5O','DRZQNzv5O',
                'JRZQNzvvO','QRZQNzvvO','DRZQNzvvO',
                ]:
        plottitle='UGM + Batch'
        #mycolor='mediumseagreen'
        #mycolor='navy'
        mycolor='C0'
        hatch='\\\\'
    # if nick in ['JRZ-Nzv1O']:
    #     plottitle='3x pool + BADGE'
    #     mycolor='blueviolet'

    if nick in [
                #'JRZ8Nzv1O','QRZ8Nzv1O','XRZ8Nzq1O',
                #'JRZ8NzviO','QRZ8NzviO','XRZ8NzqiO',
                #'JTZ8Nzv1O','QTZ8Nzv1O','XTZ8Nzq1O',
                #'JTZ8NzviO','QTZ8NzviO','XTZ8NzqiO',

                'JRZANzv1O','QRZANzv1O','DRZANzv1O',
                'JRZANzviO','QRZANzviO','DRZANzviO',
                ]:
        plottitle='All'
        mycolor='blueviolet'
    if nick in [
                #'JRZ7Nzv1O','QRZ7Nzv1O','XRZ7Nzq1O'
                'JRZDNzv1O','QRZDNzv1O','DRZDNzv1O',
                'JRZDNzviO','QRZDNzviO','DRZDNzviO',
                ]:
        plottitle='All + Batch'
        #mycolor='mediumorchid'
        #mycolor='pink'
        mycolor='blueviolet'
        #mycolor='mediumvioletred'
        hatch='\\\\'

    if nick in [
                 #'JRZCNzv1O','QRZCNzv1O','XRZCNzv1O',    
                 'JRZXNzv1O','QRZXNzv1O','DRZXNzv1O',
                 'JRZXNzviO','QRZXNzviO','DRZXNzviO',
                 ]:
        #plottitle='25K Mutagenesis + 25K Random + 10K UGM'
        plottitle='All (+60K)'
        #mycolor='chocolate'
        mycolor='violet'
    # if nick in ['JRZLNzv1O','QRZLNzv1O','XRZLNzv1O',    'JRZlNzv1O','QRZlNzv1O','XRZlNzv1O']:
    #     plottitle='25K Mutagenesis + 25K Random + 10K UGM'
    #     mycolor='saddlebrown'
    if nick in ['JRZPNzv1O','QRZPNzv1O','DRZPNzv1O',
                'JRZPNzviO','QRZPNzviO','DRZPNzviO',
                ]:
        #plottitle='(100K Mutagenesis -> 20K Batch) + (... Random -> ...) + (... UGM -> ...)'
        #plottitle='(100K UGM/Mutagenesis/Random + 20K Batch): 60K'
        plottitle='All + Batch (+60K)'
        #mycolor='maroon'
        mycolor='blueviolet'
        hatch='\\\\'

    # if request=='mutratem1O':
    #     pass
    # if request=='ivs1':
    #     # if nick in ['JRZ3NzvvO']:
    #     #     plottitle='KMeansPP (BADGE) (unanchored)'
    #     #     mycolor='olive'
    #     if nick in ['JRZ3Nzv5O']:
    #         plottitle='KMeansPP (BADGE) (anchored)'
    #         mycolor='olive'
    # if request=='vvsq':
    #     if nick in ['JRZ3Nzq5O']:
    #         plottitle='KMeansPP (BADGE) (25%)'
    #         mycolor='darkkhaki'
    #     if nick in ['JRZ3Nzv5O']:
    #         plottitle='KMeansPP (BADGE) (10%)'
    #         mycolor='olive'
    # if request=='FFig1C':
    #     if nick in ['JRZ3Nzv5O']:
    #         plottitle='KMeansPP (BADGE) (anchored)'
    #         mycolor='olive'
    if request=='quality':
        if nick in ["JRZ3Nzv5O"]: plottitle='BADGE (anchored, 5%)'
        if nick in ["JRZ1Nzv5O"]: plottitle='BatchBALD (anchored, 5%)'
        if nick in ["JRZ3Nzq5O"]: plottitle='BADGE (anchored, 25%)'
        if nick in ["JRZ1Nzq5O"]: plottitle='BatchBALD (anchored, 25%)'
        if nick in ["JTZ3Nzv5O"]: plottitle='BADGE (anchored, 5%) (Randominit.)'
        if nick in ["JTZ1Nzv5O"]: plottitle='BatchBALD (anchored, 5%) (Randominit.)'
        if nick in ["JTZ3Nzq5O"]: plottitle='BADGE (anchored, 25%) (Randominit.)'
        if nick in ["JTZ1Nzq5O"]: plottitle='BatchBALD (anchored, 25%) (Randominit.)'
        if nick in ['JRZ-Nzv1O']: plottitle='3x pool + BADGE'
        if nick in ['JRZ9Nzv5O']: plottitle='BADGE (tot.Random)'
        if nick in ['JRZ3NzveO']: plottitle='BADGE (anchored, 5%) (no repeated seq.)'
    #if request=='SI_q-v':
    if request in ['SI_5perc_vs_25perc']:
        #print("AHSA",nick)
        if nick in ["JRZJNzq1O"]: plottitle='UGM, 25%'
        if nick in ["JRZJNzv1O"]: plottitle='UGM, 5%'
        if nick in ["JRZoNzq1O"]: plottitle='Mutagenesis, 25%'
        if nick in ["JRZoNzv1O"]: plottitle='Mutagenesis, 5%'
        if nick in ["JRZJNzqiO"]: plottitle='UGM, 25%'
        if nick in ["JRZJNzviO"]: plottitle='UGM, 5%'
        if nick in ["JRZoNzqiO"]: plottitle='Mutagenesis, 25%'
        if nick in ["JRZoNzviO"]: plottitle='Mutagenesis, 5%'
        #
        if nick in ["JRZJNzq1O"]: mycolor='C9'
        if nick in ["JRZoNzq1O"]: mycolor='goldenrod'
        if nick in ["JRZJNzqiO"]: mycolor='C9'
        if nick in ["JRZoNzqiO"]: mycolor='goldenrod'
    #if request=='SI_M-O':
    if request in ['SI_Single_VS_EnsembleOracle']:
        #print("AHSBCCJSANCJSABCHSABCIUSAHCUYSABSICUNSIUCNIUSA",nick)
        if nick in ["JRZJNzv1M"]: plottitle='UGM, single oracle'
        if nick in ["JRZJNzv1O"]: plottitle='UGM, oracle ensemble'
        if nick in ["JRZJNzv1M"]: mycolor='C9'
        if nick in ["JRZJNzviM"]: plottitle='UGM, single oracle'
        if nick in ["JRZJNzviO"]: plottitle='UGM, oracle ensemble'
        if nick in ["JRZJNzviM"]: mycolor='C9'
    #if request=='SI_S-O':
    if request in ['SI_MCDropout_VS_DeepEnsemble']:
        #print("AHSBCCJSANCJSABCHSABCIUSAHCUYSABIUSAOIUCSANIUCSANICNSICUNSIUCNIUSA",nick)
        if nick in ["JRZJNzv1S"]: plottitle='UGM, Deep ensemble'
        if nick in ["JRZJNzv1O"]: plottitle='UGM, MC Dropout'
        if nick in ["JRZJNzv1S"]: mycolor='C9'
        if nick in ["JRZJNzviS"]: plottitle='UGM, Deep ensemble'
        if nick in ["JRZJNzviO"]: plottitle='UGM, MC Dropout'
        if nick in ["JRZJNzviS"]: mycolor='C9'
    
    if request=='pi1_JR':
        if nick in ["JRZJNzv1O"]: 
            plottitle='UGM (anch.)'
            mycolor='C9'
        if nick in ["JRZ4Nzv5O"]: 
            plottitle='Mutagenesis (anch.) + Batch'
            mycolor='peru'
        if nick in ["JRZANzv1O"]: 
            plottitle='All (anch.)'
            mycolor='darkmagenta'
        if nick in ["JRZoNzv1O"]: 
            plottitle='Mutagenesis (anch.)'
            mycolor='orange'
    
    if request in ['Price_K562_anchored','Price_HepG2_anchored','Price_DeepSTARR_anchored']:
        #if nick in ["JRZJNzviO","JRZJNzv1O"]: plottitle='UGM (+20K)'
        if nick in ["JRZJNzv1O","QRZJNzv1O","DRZJNzv1O"]: plottitle='UGM (+20K)'
    if request in ['Price_K562_notanchored','Price_HepG2_notanchored','Price_DeepSTARR_notanchored']:
        #if nick in ["JRZJNzviO","JRZJNzv1O"]: plottitle='UGM (+20K)'
        if nick in ["JRZJNzviO","QRZJNzviO","DRZJNzviO"]: plottitle='UGM (+20K)'
    if request in ['pPrice_JR','pPrice_QR','pPrice_DR']:
        #if nick in ["JRZJNzviO","JRZJNzv1O"]: plottitle='UGM (+20K)'
        if nick in ["JRZJNzviO","QRZJNzviO","DRZJNzviO"]: plottitle='UGM (+20K)'
        #
        if nick in ["JRZoNMv5O","QRZoNMv5O","DRZoNMv5O"]: plottitle='Mutagenesis, anch. (+100K)'
        if nick in ["JRZJNzv1O","QRZJNzv1O","DRZJNzv1O"]: plottitle='UGM, anch.'
        if nick in ["JRZXNzv1O","QRZXNzv1O","DRZXNzv1O"]: plottitle='All, anch. (+60K)'
        if nick in ["JRZPNzv1O","QRZPNzv1O","DRZPNzv1O"]: plottitle='All + Batch, anch. (+60k)'

    if request in ['Head2Head_K562_notanchored_ID','Head2Head_K562_notanchored_Mutagenesis','Head2Head_K562_notanchored_Random','PMTRD_JRi',
                   'Head2Head_HepG2_notanchored_ID','Head2Head_HepG2_notanchored_Mutagenesis','Head2Head_HepG2_notanchored_Random','PMTRD_QRi',
                   'Head2Head_DeepSTARR_notanchored_ID','Head2Head_DeepSTARR_notanchored_Mutagenesis','Head2Head_DeepSTARR_notanchored_Random','PMTRD_DRi',
                   ]:
        if nick in ["JRZoNzviO","QRZoNzviO","DRZoNzviO"]: plottitle='' # Mutagenesis         -> ''
        if nick in ["JRZoMzvvO","QRZoMzvvO","DRZoMzvvO"]: plottitle='Mutagenesis' # Mutagenesis + Unc.  -> 'Mutagenesis'
        if nick in ["JRZ4NzvvO","QRZ4NzvvO","DRZ4NzvvO"]: plottitle='' # Mutagenesis + Batch -> ''
        if nick in ["JRZJNzviO","QRZJNzviO","DRZJNzviO"]: plottitle='' # UGM         -> ''
        if nick in ["JRZQNzvvO","QRZQNzvvO","DRZQNzvvO"]: plottitle='UGM' # UGM + Unc.  -> 'Mutagenesis'
        if nick in ["JRZJNzviS","QRZJNzviS","DRZJNzviS"]: plottitle='' # Mutagenesis + Batch -> ''
        if nick in ["JRZtNzviO","QRZtNzviO","DRZtNzviO","JRZtNzt1O","QRZtNzt1O","DRZtNzt1O","JRZtNzviO",'QRZtNzq1O']: plottitle='' # Random         -> ''
        if nick in ["JRZtMzvvO","QRZtMzvvO","DRZtMzvvO"]: plottitle='Random' # Random + Unc.  -> 'Mutagenesis'
        if nick in ["JRZ6NzvvO","QRZ6NzvvO","DRZ6NzvvO"]: plottitle='' # Random + Batch -> ''        
        if nick in ["JRZDNzviO","QRZDNzviO","DRZDNzviO"]: plottitle='' # All + Batch -> ''        

    short_dict={nick:{'chosen_model':chosen_model, 
                'chosen_dataset':chosen_dataset, 
                'seq_method':seq_method, 
                'rank_method':converted[subsel]+'_spdes-'+spdes,
                'genU':genU,
                'bao':bao, 
                'mutrate':mtrate.replace('.','p'), 
                'plotname':plottitle, 
                'is_cost':is_cost,
                'color':mycolor,
                'hatch':hatch,
                }}
    #print(f"---> make_short_dict: {nick=} {short_dict=}")
    return short_dict

def get_outflag_for_nick(nick): # VR5sNlq1M
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()
    chosen_model, what_dataset, init_distrib, init_number, genU, trainmaxep, bao, seq_method, mtrate,subsel,spdes=get_args_for_nick(nick)
    outflag_new='Model-'+chosen_model+'_DS-'+what_dataset+'labels-seed0_'+init_distrib+'_'+init_number
    outflag_new+='_pN-'+init_number #QUIQUIURG correct?
    outflag_new+='_gU-'+genU
    outflag_new+='_mxe-'+str(trainmaxep)
    outflag_new+='_ALc-1'
    outflag_new+='_itr-retrain'
    outflag_new+='_bao-'+bao
    outflag_new+='_seqmeth-'+seq_method+'-mtrt-'+mtrate ##.replace('.','p') 
    outflag_new+='_pristmeth-ds'
    outflag_new+='_'+converted[subsel]
    outflag_new+='_spdes-'+spdes

    return outflag_new

def reverse_dict(dict,value,is_subsel=False):
    the_key=''
    for key in dict.keys():
        if dict[key]==value: 
            the_key=key
    #if dict==subsnick:
    if is_subsel:
        the_key=the_key.replace('5x','') #should work, with addition_subsnick
    return the_key

def get_old_nick_for_outflag(outflag):
    # Model-ResidualBind_DS-VTS1_rnacompete2009labels-seed0_random0_5000_pN-5000_gU-5000_mxe-300_ALc-1_itr-retrain_bao-50_seqmeth-saliency_pristmeth-ds_unc-no-0p0_div-no-0p0_hpred-no-0p0_spdes-mc_dropout_5
    # VR5slN
    #chosen_model=outflag.split('Model-')[1].split('_DS')[0]
    #chosen_dataset=outflag.split('DS-')[1].split('_pN')[0]
    #dsnick,seqnick,subsnick,dstrnick,numbernick,converted=get_nicks()
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()
    what_dataset=outflag.split('DS-')[1].split('labels')[0]
    init_distrib=outflag.split('seed0_')[1].split('_pN')[0].split('_')[0]
    init_number=outflag.split('seed0_')[1].split('_pN')[0].split('_')[1]
    bao=outflag.split('bao-')[1].split('_seqmeth-')[0]
    if bao=='10':
        addition_subsnick='5x'
    seqmeth=outflag.split('seqmeth-')[1].split('_pristmeth-')[0]
    subsel='unc-'+outflag.split('unc-')[1].split('_spdes-')[0]
    nick=reverse_dict(dsnick,what_dataset)+reverse_dict(dstrnick,init_distrib)+reverse_dict(numbernick,init_number)+reverse_dict(seqnick,seqmeth)+reverse_dict(subsnick,reverse_dict(converted,subsel,is_subsel=True))+addition_subsnick
    return nick

def get_nick_for_outflag(outflag): #VR5sNlq1
    # Model-ResidualBind_DS-VTS1_rnacompete2009labels-seed0_random0_5000_pN-5000_gU-5000_mxe-300_ALc-1_itr-retrain_bao-50_seqmeth-saliency_pristmeth-ds_unc-no-0p0_div-no-0p0_hpred-no-0p0_spdes-mc_dropout_5
    # VR5slN
    chosen_model=outflag.split('Model-')[1].split('_DS')[0]
    chosen_dataset=outflag.split('DS-')[1].split('_pN')[0]
    #dsnick,seqnick,subsnick,dstrnick,numbernick,converted=get_nicks()
    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()
    what_dataset=outflag.split('DS-')[1].split('labels')[0]
    init_distrib=outflag.split('seed0_')[1].split('_pN')[0].split('_')[0]
    init_number=outflag.split('seed0_')[1].split('_pN')[0].split('_')[1]
    bao=outflag.split('bao-')[1].split('_seqmeth-')[0]
    seqmeth=outflag.split('seqmeth-')[1].split('-mtrt-')[0]
    mtrate=outflag.split('-mtrt-')[1].split('_pristmeth-')[0].replace('p','.')
    subsel='unc-'+outflag.split('unc-')[1].split('_spdes-')[0]
    spdes=outflag.split('_spdes-')[1]  #4deepens

    #ntomake=1
    generatedU=outflag.split('_gU-')[1].split('_mxe-')[0] #NTOMAKELARGER
    pristineN=outflag.split('_pN-')[1].split('_gU-')[0] #NTOMAKELARGER
    #nrounds=int(int(generatedU.replace('notanch-',''))/int(pristineN)) #NTOMAKELARGER #QUIQUIURG Im not sure it should be /pristineN # goodold until 3 apr 2024
    #nrounds=int(int(generatedU.replace('notanch-',''))/int(init_number)) #great pre 18 Oct 2024
    nrounds=int(int(generatedU.replace('notanch-','').replace('screm-',''))/int(init_number)) 
    #if nrounds==1: #NTOMAKELARGER #pre anchored 31 mar 2024
    #    ntomake='same' #NTOMAKELARGER
    #elif nrounds==5: #NTOMAKELARGER
    #    ntomake='five' #NTOMAKELARGER
    #if nrounds==1 and 'notanch-' not in generatedU: #NTOMAKELARGER #great pre 18 Oct 2024
    if nrounds==1 and 'notanch-' not in generatedU and 'screm-' not in generatedU: #NTOMAKELARGER 
        ntomake='same' #NTOMAKELARGER
    #elif nrounds==5 and 'notanch-' not in generatedU: #NTOMAKELARGER #great pre 18 Oct 2024
    elif nrounds==5 and 'notanch-' not in generatedU and 'screm-' not in generatedU: #NTOMAKELARGER 
        ntomake='five' #NTOMAKELARGER
    #elif nrounds==10 and 'notanch-' not in generatedU: #NTOMAKELARGER #great pre 18 Oct 2024
    elif nrounds==10 and 'notanch-' not in generatedU and 'screm-' not in generatedU: #NTOMAKELARGER 
        ntomake='ten' #NTOMAKELARGER
    #elif nrounds==25 and 'notanch-' not in generatedU: #NTOMAKELARGER #great pre 18 Oct 2024
    elif nrounds==25 and 'notanch-' not in generatedU and 'screm-' not in generatedU: #NTOMAKELARGER 
        ntomake='twentyfaiv' #NTOMAKELARGER
    #elif nrounds==6 and 'notanch-' not in generatedU: #NTOMAKELARGER #great pre 18 Oct 2024
    elif nrounds==6 and 'notanch-' not in generatedU and 'screm-' not in generatedU: #NTOMAKELARGER 
        ntomake='six' #NTOMAKELARGER
    elif nrounds==1 and 'notanch-' in generatedU: #NTOMAKELARGER 
        ntomake='na_1' #NTOMAKELARGER
    elif nrounds==5 and 'notanch-' in generatedU: #NTOMAKELARGER 
        ntomake='na_5' #NTOMAKELARGER
    elif nrounds==5 and 'screm-' in generatedU: #NTOMAKELARGER 
        ntomake='scr5' #NTOMAKELARGER
    elif nrounds==1 and 'screm-' in generatedU: #NTOMAKELARGER 
        ntomake='scr1' #NTOMAKELARGER
    #elif nrounds==0 and 'notanch-' not in generatedU:  #great pre 18 Oct 2024
    elif nrounds==0 and 'notanch-' not in generatedU and 'screm-' not in generatedU:  
        #if float(int(generatedU.replace('notanch-','')))/int(init_number)==0.2:
        #if int(init_number)/int(generatedU.replace('notanch-',''))==5: #great pre 18 Oct 2024
        if int(init_number)/int(generatedU.replace('notanch-','').replace('screm-',''))==5:
            ntomake='fifth'
        #elif int(init_number)/int(generatedU.replace('notanch-',''))==6: #great pre 18 Oct 2024
        elif int(init_number)/int(generatedU.replace('notanch-','').replace('screm-',''))==6:
            ntomake='sixth'
        #elif int(init_number)/int(generatedU.replace('notanch-',''))==13: #great pre 18 Oct 2024
        elif int(init_number)/int(generatedU.replace('notanch-','').replace('screm-',''))==13:
            ntomake='thirteenth'
        #elif int(init_number)/int(generatedU.replace('notanch-',''))==30: #great pre 18 Oct 2024
        elif int(init_number)/int(generatedU.replace('notanch-','').replace('screm-',''))==30:
            ntomake='dreizig'
        #elif int(init_number)/int(generatedU.replace('notanch-',''))==2: #great pre 18 Oct 2024
        elif int(init_number)/int(generatedU.replace('notanch-','').replace('screm-',''))==2:
            ntomake='half'
    else:
        print(f"ERROR: nrounds must be wrong: {generatedU=} {pristineN=} {nrounds=}")
    
    if 'LentiMPRA_' in chosen_dataset:
        if chosen_model=='LegNet':
            reverseddict_dsnick='G'
        elif chosen_model=='LegNet_Custom':
            reverseddict_dsnick='P'
        elif chosen_model=='NewResNet':
            reverseddict_dsnick='K'
        elif chosen_model=='mpra':
            reverseddict_dsnick='m'
        elif chosen_model=='ResidualBind':
            reverseddict_dsnick='L'
        else:
            print("ERROR model chosen")
            exit()
    
    if 'newLentiMPRAK562_' in chosen_dataset and chosen_model=='NewResNet':
        reverseddict_dsnick='K'
    elif 'newLentiMPRAHepG2_' in chosen_dataset and chosen_model=='NewResNet':
        reverseddict_dsnick='H'
    elif 'newLentiMPRAK562two_' in chosen_dataset and chosen_model=='NewResNet':
        reverseddict_dsnick='a'
    elif 'newLentiMPRAHepG2two_' in chosen_dataset and chosen_model=='NewResNet':
        reverseddict_dsnick='b'

    elif 'newLentiMPRAK562_' in chosen_dataset and chosen_model=='LegNet_Custom':
        reverseddict_dsnick='J'
    elif 'newLentiMPRAHepG2_' in chosen_dataset and chosen_model=='LegNet_Custom':
        reverseddict_dsnick='Q'
    elif 'newLentiMPRAK562two_' in chosen_dataset and chosen_model=='LegNet_Custom':
        reverseddict_dsnick='c'
    elif 'newLentiMPRAHepG2two_' in chosen_dataset and chosen_model=='LegNet_Custom':
        reverseddict_dsnick='d'

    else:
        reverseddict_dsnick=reverse_dict(dsnick,what_dataset)
    #print(reverse_dict(dsnick,what_dataset))
    #print(reverse_dict(dstrnick,init_distrib))
    #print(reverse_dict(numbernick,init_number))
    #print(reverse_dict(seqnicknew,seqmeth))
    #print(reverse_dict(subsnick,reverse_dict(converted,subsel)))
    #print(reverse_dict(baonick,bao))
    #print(mtrtnick,mtrate,reverse_dict(mtrtnick,mtrate))
    #print(n2mnick,reverse_dict(n2mnick,'1'))
    """
    nick=reverse_dict(dsnick,what_dataset)+\
         reverse_dict(dstrnick,init_distrib)+\
         reverse_dict(numbernick,init_number)+\
         reverse_dict(seqnicknew,seqmeth)+\
         reverse_dict(subsnick,reverse_dict(converted,subsel)).replace('5x','')+\
         reverse_dict(baonick,bao)+\
         reverse_dict(mtrtnick,mtrate)+\
         reverse_dict(n2mnick,ntomake)+\
         reverse_dict(spdesnick,spdes) #4deepens
    """
    nick=reverseddict_dsnick+\
         reverse_dict(dstrnick,init_distrib)+\
         reverse_dict(numbernick,init_number)+\
         reverse_dict(seqnicknew,seqmeth)+\
         reverse_dict(subsnick,reverse_dict(converted,subsel)).replace('5x','')+\
         reverse_dict(baonick,bao)+\
         reverse_dict(mtrtnick,mtrate)+\
         reverse_dict(n2mnick,ntomake)+\
         reverse_dict(spdesnick,spdes) #4deepens
    #QUIQUIURG reverse_dict(n2mnick,ntomake)+\  has something to inspect
    #reverse_dict(dsnick,what_dataset) is the problematic line for LegNet
    return nick


if __name__=='__main__':
    args = parser.parse_args()

    dsnick,seqnick,subsnick,dstrnick,numbernick,converted,mtrtnick,baonick,seqnicknew,n2mnick,spdesnick=get_nicks()

    if args.old=='1':
        what_dataset=dsnick[args.nick[0]]
        init_distrib=dstrnick[args.nick[1]]
        init_number=numbernick[args.nick[2]]
        subsel=subsnick[args.nick[5:]]
        seq_method=seqnick[args.nick[3:5]]
    else:
        what_dataset=dsnick[args.nick[0]] #V
        init_distrib=dstrnick[args.nick[1]] #R
        init_number=numbernick[args.nick[2]] #5
        subsel=subsnick[args.nick[4]] #N
        seq_method=seqnicknew[args.nick[3]] #s
        #
        bao=baonick[args.nick[5]] #l
        mtrate=mtrtnick[args.nick[6]] #q
        n2make=n2mnick[args.nick[7]] #1
        
        #spdes='mc_dropout_5' 
        spdes=spdesnick[args.nick[8]] #M #4deepens

    if spdes=='sigma_deep_ensemble':
        Nmodels='5'
    elif spdes=='sigma_deep_ensemble0deepens':
        Nmodels='5'
    else:
        Nmodels='1'

    if args.nick[0]=='V':
        chosen_model='ResidualBind'
    elif args.nick[0]=='T':
        chosen_model='ResidualBind'
    elif args.nick[0]=='X':
        chosen_model='ResidualBind'
    elif args.nick[0]=='D':
        chosen_model='DeepSTARR'
    elif args.nick[0]=='L':
        chosen_model='ResidualBind'
    elif args.nick[0]=='G':
        chosen_model='LegNet'
    elif args.nick[0]=='P':
        chosen_model='LegNet_Custom'
    
    elif args.nick[0]=='K':
        chosen_model='NewResNet'
    elif args.nick[0]=='a':
        chosen_model='NewResNet'
    elif args.nick[0]=='H':
        chosen_model='NewResNet'
    elif args.nick[0]=='b':
        chosen_model='NewResNet'
    
    elif args.nick[0]=='J':
        chosen_model='LegNet_Custom'
    elif args.nick[0]=='c':
        chosen_model='LegNet_Custom'
    elif args.nick[0]=='Q':
        chosen_model='LegNet_Custom'
    elif args.nick[0]=='d':
        chosen_model='LegNet_Custom'

    elif args.nick[0]=='m':
        chosen_model='mpra'
    else:
        print("ERROR: wrong dataset choice.")
        exit()

    #if [ $subsnick = N5x ] ; then 
    #subsel=no-0.0-no-0.0-no-0.0 
    #genU=5000
    #hmnbao=10
    #fi
    #if [ $subsnick = M5x ] ; then 
    #subsel=mc_dropout_5-1.0-no-0.0-no-0.0 
    #genU=5000
    #hmnbao=10
    #fi

    #if [ $numbernick = O ] ; then 
    #number=1000 
    #genU=1000
    #fi
    #if [ $numbernick = T ] ; then 
    #number=10000 
    #genU=10000
    #fi 

    #genU='5000'
    #if init_number!='5000': #B2
    #    genU=init_number
    
    if args.nick[7]=='i':
        genU='notanch-'+init_number #related to n2make
    elif args.nick[7]=='v':
        genU='notanch-'+str(int(init_number)*5) #related to n2make
    elif args.nick[7]=='a':
        genU='screm-'+str(int(init_number)*1) #related to n2make
    elif args.nick[7]=='e':
        genU='screm-'+str(int(init_number)*5) #related to n2make
    elif args.nick[7]=='f':
        genU=str(int(int(init_number)/5))
    elif args.nick[7]=='T':
        genU='50000' #str(int(int(init_number)/10)): AC: this will give 500, which I dont want for LR5FMqqTM
        print("WARNING: this is tweaked specifically for LR5FMqqTM")
    elif args.nick[7]=='X':
        genU='125000' #str(int(int(init_number)/10)): AC: this will give 500, which I dont want for LR5FMqqTM
        print("WARNING: this is tweaked specifically for LR5FMqqXM")
    elif args.nick[7]=='s':
        genU=str(int(int(init_number)/6))
    elif args.nick[7]=='D':
        genU=str(int(int(init_number)/30))
    elif args.nick[7]=='p':
        genU=str(int(int(init_number)/13))
    elif args.nick[7]=='h':
        genU=str(int(int(init_number)/2))
    else:
        #genU=str(5000*int(args.nick[7])) #related to n2make
        genU=str(int(init_number)*int(args.nick[7])) #related to n2make


    if args.old=='1':
        hmnbao='50'
        if '5x' in args.nick[5:]: #A2
            hmnbao='10'

        if '5x' in args.nick[5:] and init_number!='5000':
            print("WARNING: mixing up B2 and A2")
            exit()
    else:
        hmnbao=bao

    pre_sel='no' #A3

    if args.old=='1':
        mutrate='0.25'
    else:
        mutrate=mtrate ##.replace('p','.')

    #VR5mtN
    #print("./Launch_DAL.sh ResidualBind "$dsname"labels-seed"$ds_seed"_"$distrib"_"$number" $subsel $seqmeth 1 $genU $fullnick 100 mc_dropout_5 $pre_sel $hmnbao | grep qsub >> $qsbf")
    
    """
    print()
    print(f"{what_dataset=}")
    print(f"{init_distrib=}")
    print(f"{init_number=}")
    print(f"{subsel=}")
    print(f"{seq_method=}")
    print(f"{genU=}")
    print(f"{pre_sel=}")
    print(f"{hmnbao=}")
    print()
    """
    #print("./Launch_DAL.sh ResidualBind "+what_dataset+"labels-seed0_"+init_distrib+"_"+init_number+" "+subsel+" "+seq_method+" 1 "+genU+" "+args.nick+" 100 mc_dropout_5 "+pre_sel+" "+hmnbao+" | grep qsub")
    print("#./Launch_DAL.sh CHOSEN_MODEL WHAT_DATASET+labels-seed0_+INIT_DISTRIB+_+INIT_NUMBER SUBSEL SEQ_METHOD 1 GENU NICK 100 mc_dropout_5 PRE_SEL HMNBAO MUTRATE | grep qsub")
    #print("./Launch_DAL.sh "+chosen_model+" "+what_dataset+"labels-seed0_"+init_distrib+"_"+init_number+" "+subsel+" "+seq_method+" 1 "+genU+" "+args.nick+" 100 mc_dropout_5 "+pre_sel+" "+hmnbao+" "+mutrate+" | grep qsub")
    print("./Launch_DAL.sh "+chosen_model+" "+what_dataset+"labels-seed0_"+init_distrib+"_"+init_number+" "+subsel+" "+seq_method+" "+Nmodels+" "+genU+" "+args.nick+" 100 "+spdes+" "+pre_sel+" "+hmnbao+" "+mutrate+" | grep qsub")
    #print()







# NICK!!!!!!!!!!!! ---> outflag
# VR5slN5x
#grep [5:]

# VR5slN5xq
# VR5slNlq: the new standard
# VR5sNlq: ?
# required another : n_to_make? factor?
# VR5sNlq1: ?
# if two sp