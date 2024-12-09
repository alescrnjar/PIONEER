import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='get_nickloop',type=str)
parser.add_argument('--exp',default='FFig1i',type=str)

#def placeholder_nick_manual(nick):
def placeholder_nick(nick, verbose=True):
    # totally_random and fromfile do not depend on mutation rate (q,t,v), and also do not depend on anchoring (1/i,5/v)
    placeholders={
        'JRZtNMv5O':'JRZtNMt5O','JRZtNzv1O':'JRZtNzt1O', 'JRZtNMvvO':'JRZtNMt5O','JRZtNzviO':'JRZtNzt1O',
        'JLZtNzv1O':'JLZtNzq1O','JLZtNzviO':'JLZtNzq1O', 'JLZtNMv5O':'JLZtNMq5O','JLZtNMvvO':'JLZtNMq5O', 
        'JRZFNzv1O':'JRZFNzq1O','JRZFNzviO':'JRZFNzq1O',
        
        'JRZdNzq1O':'JRZdNzv1O',
        'JRZdNzviO':'JRZdNzv1O',
        'QRZdNzq1O':'QRZdNzv1O',
        'QRZdNzviO':'QRZdNzv1O',


        'QRZtNztiO':'QRZtNzq1O',
        'QRZFNztiO':'QRZFNzt1O',
        'XRZtNztiO':'XRZtNzq1O',
        'XRZFNztiO':'XRZFNzt1O',
        'JRZtNztiO':'JRZtNzt1O',
        'JTZtNztiO':'JTZtNzv1O',
        'QRZtNMtvO':'QRZtNMq5O',
        'XRZtNMtvO':'XRZtNMq5O',

        'XRZFNzq1O':'XRZFNzt1O',
        'JTZtNzviO':'JTZtNzv1O',
        'QRZFNzviO':'QRZFNzv1O',

        }
    if nick not in placeholders.keys():
        fakenick=nick
    else:
        fakenick=placeholders[nick]
    return fakenick

# def placeholder_nick(nick, verbose=True):
#     if os.uname()[1]=='amethyst' or os.uname()[1]=='auros':
#         return placeholder_nick_manual(nick)
#     else:
#         anchvariants={'1':['1','i'],'5':['5','v'],'i':['1','i'],'v':['5','v']}
#         fakenick=nick
#         if nick[3]=='t' or nick[3]=='F':
#             any_found=False
#             for mutlett in ['v','q','t']:
#                 for anch in anchvariants[nick[-2]]:
#                     #nick1=nick[:6]+mutlett+nick[7:]
#                     nick1=nick[:6]+mutlett+anch+nick[-1]
#                     wcl0=int(os.popen("grep BASH /grid/koo/home/crnjar/Occasio_Dev/noh*$nick*Job* | wc -l").read())
#                     wcl1=int(os.popen("grep BASH /grid/koo/home/crnjar/outputs_DALdna_4plot/noh*$nick*Job* | wc -l").read())
#                     #if wcl0!=0 or wcl1!=0:
#                     if wcl0==5 or wcl1==5:
#                         if verbose: print("Assigned",nick1,"instead of",nick)
#                         fakenick=nick1
#                         #any_found=True
#             if verbose and not any_found: print("No placeholders automatically assigned for",nick) ####,"(in principle you should never see this as it covers even the self identical case)")
#         else:
#             fakenick=nick
#         return fakenick
def find_placeholder(nick, verbose=True):
    if os.uname()[1]=='amethyst' or os.uname()[1]=='auros':
        #return placeholder_nick_manual(nick)
        return placeholder_nick(nick)
    else:
        anchvariants={'1':['1','i'],'5':['5','v'],'i':['1','i'],'v':['5','v']}
        fakenick=nick
        if nick[3]=='t' or nick[3]=='F':
            wcl0_current=int(os.popen("grep BASH /grid/koo/home/crnjar/Occasio_Dev/noh*"+nick+"*Job* 2> prov | wc -l").read())
            wcl1_current=int(os.popen("grep BASH /grid/koo/home/crnjar/outputs_DALdna_4plot/noh*"+nick+"*Job* 2> prov  | wc -l").read())      
            if wcl0_current==0 and wcl1_current==0:
                any_found=False
                for mutlett in ['v','q','t']:
                    for anch in anchvariants[nick[-2]]:
                        #nick1=nick[:6]+mutlett+nick[7:]
                        nick1=nick[:6]+mutlett+anch+nick[-1]
                        wcl0=int(os.popen("grep BASH /grid/koo/home/crnjar/Occasio_Dev/noh*"+nick1+"*Job* 2> prov  | wc -l").read())
                        wcl1=int(os.popen("grep BASH /grid/koo/home/crnjar/outputs_DALdna_4plot/noh*"+nick1+"*Job* 2> prov  | wc -l").read())
                        if wcl0!=0 or wcl1!=0:
                        #if wcl0==5 or wcl1==5:
                            if verbose: print("Assigned",nick1,"instead of",nick)
                            fakenick=nick1
                            #any_found=True
                if verbose and not any_found: print("No placeholders automatically assigned for",nick) ####,"(in principle you should never see this as it covers even the self identical case)")
        else:
            fakenick=nick
        return fakenick

    
def get_nick_dictionary(verbose=True):
    if verbose: print("Experiments.py: getting nick dictionary...")
    all_nicks={        
        'Head2Head_K562_notanchored_ID': ["JRZFNzviO",       "JRZoNzviO","JRZoMzvvO","JRZ4NzvvO",         "JRZJNzviO","JRZQNzvvO","JRZJNzviS",      "JRZtNzviO","JRZtMzvvO","JRZ6NzvvO",      "JRZANzviO","JRZDNzviO"], # 'PID_JRi'
        'Head2Head_HepG2_notanchored_ID': ["QRZFNzviO",       "QRZoNzviO","QRZoMzvvO","QRZ4NzvvO",         "QRZJNzviO","QRZQNzvvO","QRZJNzviS",      "QRZtNzviO","QRZtMzvvO","QRZ6NzvvO",      "QRZANzviO","QRZDNzviO"], # 'PID_QRi'
        'Head2Head_DeepSTARR_notanchored_ID': ["DRZFNzviO",       "DRZoNzviO","DRZoMzvvO","DRZ4NzvvO",         "DRZJNzviO","DRZQNzvvO","DRZJNzviS",      "DRZtNzviO","DRZtMzvvO","DRZ6NzvvO",      "DRZANzviO","DRZDNzviO"], # 'PID_DRi'

        'Head2Head_K562_notanchored_Mutagenesis': ["JRZFNzviO",       "JRZoNzviO","JRZoMzvvO","JRZ4NzvvO",         "JRZJNzviO","JRZQNzvvO","JRZJNzviS",      "JRZtNzviO","JRZtMzvvO","JRZ6NzvvO",      "JRZANzviO","JRZDNzviO"], # 'PMT_JRi'
        'Head2Head_HepG2_notanchored_Mutagenesis': ["QRZFNzviO",       "QRZoNzviO","QRZoMzvvO","QRZ4NzvvO",         "QRZJNzviO","QRZQNzvvO","QRZJNzviS",      "QRZtNzviO","QRZtMzvvO","QRZ6NzvvO",      "QRZANzviO","QRZDNzviO"], # 'PMT_QRi'
        'Head2Head_DeepSTARR_notanchored_Mutagenesis': ["DRZFNzviO",       "DRZoNzviO","DRZoMzvvO","DRZ4NzvvO",         "DRZJNzviO","DRZQNzvvO","DRZJNzviS",      "DRZtNzviO","DRZtMzvvO","DRZ6NzvvO",      "DRZANzviO","DRZDNzviO"], # 'PMT_DRi'
        
        'Head2Head_K562_notanchored_Random': ["JRZFNzviO",       "JRZoNzviO","JRZoMzvvO","JRZ4NzvvO",         "JRZJNzviO","JRZQNzvvO","JRZJNzviS",      "JRZtNzviO","JRZtMzvvO","JRZ6NzvvO",      "JRZANzviO","JRZDNzviO"], # 'PRD_JRi'
        'Head2Head_HepG2_notanchored_Random': ["QRZFNzviO",       "QRZoNzviO","QRZoMzvvO","QRZ4NzvvO",         "QRZJNzviO","QRZQNzvvO","QRZJNzviS",      "QRZtNzviO","QRZtMzvvO","QRZ6NzvvO",      "QRZANzviO","QRZDNzviO"], # 'PRD_QRi'
        'Head2Head_DeepSTARR_notanchored_Random': ["DRZFNzviO",       "DRZoNzviO","DRZoMzvvO","DRZ4NzvvO",         "DRZJNzviO","DRZQNzvvO","DRZJNzviS",      "DRZtNzviO","DRZtMzvvO","DRZ6NzvvO",      "DRZANzviO","DRZDNzviO"], # 'PRD_DRi'
       
        # 'pDShiftGlobloc':["JRZFNzviO",     "JRZJNzviO","JRZJNzv1S",     "JRZ6NzvvO",       "JRZANzviO","JRZDNzviO"],   # The only one to show as global local
        
        'Price_K562_notanchored':["JRZtNMvvO", "JRZoNMvvO", "JRZJNzviO", "JRZXNzviO"], #,"JRZPNzviO"], #pPrice_JRi
        'Price_K562_anchored':["JRZtNMv5O", "JRZoNMv5O", "JRZJNzv1O", "JRZXNzv1O"], #,"JRZPNzv1O"], #pPrice_JR1
        'Price_HepG2_notanchored':["QRZtNMvvO", "QRZoNMvvO", "QRZJNzviO", "QRZXNzviO"], #,"QRZPNzviO"], #pPrice_QRi
        'Price_HepG2_anchored':["QRZtNMv5O", "QRZoNMv5O", "QRZJNzv1O", "QRZXNzv1O"], #,"QRZPNzv1O"], #pPrice_QR1
        'Price_DeepSTARR_notanchored':["DRZtNMvvO", "DRZoNMvvO", "DRZJNzviO", "DRZXNzviO"], #,"DRZPNzviO"], #pPrice_DRi
        'Price_DeepSTARR_anchored':["DRZtNMv5O", "DRZoNMv5O", "DRZJNzv1O", "DRZXNzv1O"], #,"DRZPNzv1O"], #pPrice_DR1
        
        'Head2Head_K562_anchored_vs_notanchored':["JRZJNzviO","JRZJNzv1O","JRZ4NzvvO","JRZ4Nzv5O",         "JRZoNzviO","JRZoNzv1O" ], #,         "JRZ6Nzv5O","JRZ6NzvvO"], # pi1_JR
        'Head2Head_HepG2_anchored_vs_notanchored':["QRZJNzviO","QRZJNzv1O","QRZ4NzvvO","QRZ4Nzv5O",         "QRZoNzviO","QRZoNzv1O" ], #,         "JRZ6Nzv5O","JRZ6NzvvO"], # pi1_QR
        'Head2Head_DeepSTARR_anchored_vs_notanchored':["DRZJNzviO","DRZJNzv1O","DRZ4NzvvO","DRZ4Nzv5O",         "DRZoNzviO","DRZoNzv1O" ], #,         "JRZ6Nzv5O","JRZ6NzvvO"], # pi1_DR
        

        'SI_singleoracle_vs_oracleenseemble_K562': ["JRZJNzviM","JRZJNzviO"], # SI_M-O
        'SI_mcdropout_vs_deepensemble_K562': ["JRZJNzviS","JRZJNzviO"], # SI_D-O
        'SI_25percent_vs_5percent_K562': ["JRZJNzqiO","JRZJNzviO","JRZoNzqiO","JRZoNzviO"], # SI_q-v

        'SI_singleoracle_vs_oracleenseemble_HepG2': ["QRZJNzviM","QRZJNzviO"], # SI_M-O
        'SI_mcdropout_vs_deepensemble_HepG2': ["QRZJNzviS","QRZJNzviO"], # SI_D-O
        'SI_25percent_vs_5percent_HepG2': ["QRZJNzqiO","QRZJNzviO","QRZoNzqiO","QRZoNzviO"], # SI_q-v

        'SI_singleoracle_vs_oracleenseemble_DeepSTARR': ["DRZJNzviM","DRZJNzviO"], # SI_M-O
        'SI_mcdropout_vs_deepensemble_DeepSTARR': ["DRZJNzviS","DRZJNzviO"], # SI_D-O
        'SI_25percent_vs_5percent_DeepSTARR': ["DRZJNzqiO","DRZJNzviO","DRZoNzqiO","DRZoNzviO"], # SI_q-v

        # #####

             }
    all_nicks_with_placeholders={}
    for key in all_nicks.keys():
        all_nicks_with_placeholders[key]=[placeholder_nick(nick, verbose) for nick in all_nicks[key]]
    if verbose: print("Experiments.py: got nick dictionary.")
    return all_nicks_with_placeholders

def get_nickloop(exp, verbose=True):
    return list(get_nick_dictionary(verbose)[exp])
def get_nickloop_bash(exp,verbose=False):
    return ' '.join(get_nick_dictionary(verbose)[exp])

def get_experiments(verbose=True):
    return list(get_nick_dictionary(verbose).keys())
def get_experiments_bash(verbose=False):
    return ' '.join(get_nick_dictionary(verbose).keys())

def every_possible_nick(verbose=True):
    all_nicks=[]
    for exp in get_nick_dictionary(verbose):
        #print(exp)
        for nick in get_nick_dictionary(verbose)[exp]:
            if nick not in all_nicks: all_nicks.append(nick)
    return all_nicks

def every_possible_nick_bash(verbose=False):
    return ' '.join(every_possible_nick(verbose))

def get_titles():
    titles={
        # 'FFig11':'5% Anchored',
        # 'FFig1i':'5% Unanchored',
        # 'ivs1':'5%, Anchored VS Unanch.',
        # 'vvsq':'5% VS 25%, Anchored',
        # 'FFig1q':'25% Anchored',
        # 'FFig1C':'Cost-aware comparison',
        # 'FFig1T':'Initial DS as Random sequences',

        'FJig11':'5% Anchored',
        'FJig1i':'5% Unanchored',
        'Jivs1':'5%, Anchored VS Unanch.',
        'Jvvsq':'5% VS 25%, Anchored',
        'FJig1q':'25% Anchored',
        'FJig1C':'Cost-aware comparison',
        'FJig1T':'Initial DS as Random sequences',

        'FQig11':'5% Anchored',
        'FQig1i':'5% Unanchored',
        'Qivs1':'5%, Anchored VS Unanch.',
        'Qvvsq':'5% VS 25%, Anchored',
        'FQig1q':'25% Anchored',
        'FQig1C':'Cost-aware comparison',
        'FQig1T':'Initial DS as Random sequences',

        'FXig11':'5% Anchored',
        'FXig1i':'5% Unanchored',
        'Xivs1':'5%, Anchored VS Unanch.',
        'Xvvsq':'5% VS 25%, Anchored',
        'FXig1q':'25% Anchored',
        'FXig1C':'Cost-aware comparison',
        'FXig1T':'Initial DS as Random sequences',

                }
    return titles 

def get_undesirable_seeds(nick):
    undes={
        'JRZFNzviO':[]
    }
    if nick not in list(undes.keys()):
        return []
    else:
        return undes[nick]

if __name__=='__main__':
    args = parser.parse_args()
    if args.mode=='get_experiments':
        print(get_experiments_bash())
    # for exp in get_nick_dictionary().keys():
    #     nickloop=get_nickloop(exp)
    #     print(nickloop)
    if args.mode=='get_nickloop':
        print(get_nickloop_bash(args.exp))
    if args.mode=='every_possible_nick':
        print(every_possible_nick_bash())
    if args.mode=='find_placeholders':
        for nick in every_possible_nick(verbose=False):
            fakenick=find_placeholder(nick, verbose=False)
            if fakenick!=nick: print("'"+nick+"':'"+fakenick+"',")
