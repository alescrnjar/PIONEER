import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import os

# https://github.com/p-koo/residualbind/blob/master/residualbind.py
# RNAcompete-S dataset: https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete-S/index.html

class per_factor(nn.Module):
    def __init__(self, num_filters, filter_size, f):
    #def __init__(self, num_filters, filter_size, f, activation='ReLU'): #TRUESOFT+
        super(per_factor, self).__init__()
        self.activation=nn.ReLU()
        #if activation=='ReLU': #TRUESOFT+
        #    self.activation = nn.ReLU()
        #elif activation=='softplus_beta1': #TRUESOFT+
        #    self.activactivation = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20)
        self.dropout=nn.Dropout(0.1)
        self.conv=nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, stride=1, bias=False,padding='same',dilation=f)
        self.bnorm=nn.BatchNorm1d(num_filters)

    def forward(self, x):
        out=x
        out=self.activation(out)
        out=self.dropout(out)
        out=self.conv(out)
        out=self.bnorm(out)
        return out

class residual_block(nn.Module):
    #def __init__(self, input_layer, input_shape, filter_size, activation='relu', dilated=False):
    #def __init__(self, input_shape1, filter_size, activation='relu', dilated=False):
    def __init__(self, num_filters, filter_size, activation='ReLU', dilated=False):
        super(residual_block, self).__init__()
        if dilated:
            factor = [2, 4, 8]
        else:
            factor = [1]
        #num_filters = input_layer.shape.as_list()[-1]              
        #num_filters = input_layer.size(1)
        #num_filters=input_shape1
        self.conv = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, bias=False,padding='same',dilation=1)
        self.batchnorm = nn.BatchNorm1d(num_filters)

        per_factor_s=[]
        for i_f,f in enumerate(factor):
            #if i_f==0:
            #    per_factor_s.append(per_factor(96, filter_size, f))
            #else:
            #    per_factor_s.append(per_factor(num_filters, filter_size, f))
            per_factor_s.append(per_factor(num_filters, filter_size, f))
            #per_factor_s.append(per_factor(num_filters, filter_size, f, activation)) #TRUESOFT+
        self.per_factor_s=nn.ModuleList(per_factor_s) # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        
        if activation=='ReLU': # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
            self.final_activ = nn.ReLU() # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
        elif activation=='softplus_beta1': # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
            self.final_activ = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20) # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
        #nn = keras.layers.add([input_layer, nn]) # https://www.tensorflow.org/api_docs/python/tf/keras/layers/add
        #return keras.layers.Activation(activation)(nn) #TRUESOFT+: there was a missing final activation here!!!

    def forward(self, x):
        out=x
        out=self.conv(out)
        out=self.batchnorm(out)
        for i_f in range(len(self.per_factor_s)):
            out=self.per_factor_s[i_f](out)
        out=out+x # AC: ResidualBind has a single (credo) skip connection
        out=self.final_activ(out) # standardrelu1 TRUESOFT+?
        return out

from daedalus_model import *
class ResidualBind_AC(nn.Module):
    #def __init__(self, input_shape=(41,4), num_class=1, weights_path='./weights.hdf5', classification=False, with_residual=True):
    #def __init__(self, input_shape=(4,41), num_class=1, weights_path='./weights.hdf5', classification=False, with_residual=True):
    def __init__(self, input_shape=(41,4), num_class=1, classification=False, with_residual=True, activation='ReLU', wanted_initial_attention=False, wanted_hook=False):
        super(ResidualBind_AC, self).__init__()
        self.input_shape = input_shape
        self.num_class = num_class
        #self.weights_path = weights_path
        self.classification = classification
        ##self.model = self.build(input_shape)

        self.wanted_initial_attetion=wanted_initial_attention
        if wanted_initial_attention:
            self.pe=PositionalEncoding(4, input_shape[0])
            self.attention=SelfAttention(input_shape[0])
        
        # layer 1
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=96, kernel_size=11, stride=1, bias=False, padding='same') # padding=5)
        #self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=96, kernel_size=11, stride=1, bias=False, padding='same') # padding=5)
        self.bn = nn.BatchNorm1d(96)
        if activation=='ReLU':
            self.activ = nn.ReLU()
        elif activation=='softplus_beta1':
            self.activ = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20)
        self.drop1 = nn.Dropout(0.1)

        # dilated residual block
        self.resblock=None
        num_filters = 96 #input_layer.shape.as_list()[-1]  
        #if with_residual: self.resblock = residual_block(input_shape1=input_shape[1], filter_size=3, dilated=True)
        #if with_residual: self.resblock = residual_block(input_shape1=input_shape[0], filter_size=3, dilated=True)
        if with_residual: self.resblock = residual_block(num_filters=num_filters, filter_size=3, dilated=True)
        #if with_residual: self.resblock = residual_block(num_filters=num_filters, filter_size=3, activation=activation, dilated=True) #TRUESOFT+

        # average pooling
        #self.avg_pool = F.avg_pool1d(kernel_size=10)
        self.avg_pool = nn.AvgPool1d(kernel_size=10)
        self.drop2 = nn.Dropout(0.2)

        """
        # layer 2
        """

        # Fully-connected NN
        self.flatten=nn.Flatten()
        self.dense = nn.LazyLinear(256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        if activation=='ReLU':
            self.activ2 = nn.ReLU()
        elif activation=='softplus_beta1':
            self.activ2 = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20)
        self.drop3 = nn.Dropout(0.5)

        # output layer
        self.out_layer = nn.LazyLinear(self.num_class, bias=True)

        self.wanted_hook=wanted_hook
        if wanted_hook:
            self.registered_hook = None  # Placeholder for the hook

    def forward(self, x):
        out=x

        if self.wanted_initial_attetion:
            out=self.pe(out)
            out=self.attention(out)

        #print(f"HERE {out.shape=}")
        #exit()
        out=self.conv1(out)
        #layer1=out
        #self.registered_hook = layer1.register_hook(lambda grad: setattr(self, 'first_layer_grad', grad))
        #
        #print(f"HERE self.conv1({out.shape=}")
        #exit()
        out=self.bn(out)
        #
        #out=self.activ(out)
        layer1=self.activ(out)
        if self.wanted_hook: self.registered_hook = layer1.register_hook(lambda grad: setattr(self, 'first_layer_grad', grad))
        out=layer1        
        #
        out=self.drop1(out)
        #layer1=self.drop1(out)
        #self.registered_hook = layer1.register_hook(lambda grad: setattr(self, 'first_layer_grad', grad))
        #out=layer1        

        if self.resblock!=None: 
            out=self.resblock(out)
        
        out=self.avg_pool(out)
        out=self.drop2(out)

        out=self.flatten(out)
        out=self.dense(out)
        out=self.bn2(out)
        out=self.activ2(out)
        out=self.drop3(out)

        out=self.out_layer(out)
        if self.classification: out=nn.Sigmoid(out)
        return out
    
    """
    def predict(self, X, batch_size=100, load_weights=False):
        if load_weights:
            self.load_weights()

        return self.model.predict(X, batch_size=batch_size)

    def test_model(self, test, batch_size=100, load_weights=None):
        if self.classification:
            metrics = self.model.test_model(test['inputs'], test['targets'])
        else:
            predictions = self.predict(test['inputs'], batch_size, load_weights)
            metrics = pearsonr_scores(test['targets'], predictions)
        return metrics
    """

    def initial_attention_output(self, x):
        out=x
        if self.wanted_initial_attetion:
            out=self.pe(out)
            out=self.attention(out)
        return out







#############################################################################################

"""
from tensorflow import keras
from tensorflow.keras import backend as K
#class ResidualBind():
class ResidualBind_Orig():

    def __init__(self, input_shape=(41,4), num_class=1, weights_path='.', classification=False):

        self.input_shape = input_shape
        self.num_class = num_class
        self.weights_path = weights_path
        self.classification = classification
        self.model = self.build(input_shape)


    def build(self, input_shape):
        K.clear_session()

        def residual_block(input_layer, filter_size, activation='relu', dilated=False):

            if dilated:
                factor = [2, 4, 8]
            else:
                factor = [1]
            num_filters = input_layer.shape.as_list()[-1]  

            print("--- Into Residual Block")
            print(f"R0 {num_filters=}")
            print(f"R1 {input_layer.shape=}")                            
            nn = keras.layers.Conv1D(filters=num_filters,
                                           kernel_size=filter_size,
                                           activation=None,
                                           use_bias=False,
                                           padding='same',
                                           dilation_rate=1,
                                           )(input_layer)
            print(f"R2 {nn.shape=}")                             
            nn = keras.layers.BatchNormalization()(nn)
            print(f"R3 {nn.shape=}")                            
            for f in factor:
                nn = keras.layers.Activation('relu')(nn)
                nn = keras.layers.Dropout(0.1)(nn)
                print(f"R4 {nn.shape=}")                   
                nn = keras.layers.Conv1D(filters=num_filters,
                                               kernel_size=filter_size,
                                               strides=1,
                                               activation=None,
                                               use_bias=False, 
                                               padding='same',
                                               dilation_rate=f,
                                               )(nn) 
                print(f"R5 {nn.shape=}")                            
                nn = keras.layers.BatchNormalization()(nn)
                print(f"R6 {nn.shape=}")                            
            nn = keras.layers.add([input_layer, nn])
            return keras.layers.Activation(activation)(nn)

        # input layer
        inputs = keras.layers.Input(shape=input_shape)
        print(f"{inputs.shape=}") #AC
        
        # layer 1
        nn = keras.layers.Conv1D(filters=96,
                                 kernel_size=11,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 )(inputs)   
        print(f"{nn.shape=}")                            
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        print(f"{nn.shape=}")               

        # dilated residual block
        nn = residual_block(nn, filter_size=3, dilated=True)

        # average pooling
        nn = keras.layers.AveragePooling1D(pool_size=10)(nn)
        nn = keras.layers.Dropout(0.2)(nn)

        #""
        # layer 2
        nn = keras.layers.Conv1D(filters=128,
                                 kernel_size=3,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 )(nn)                               
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = residual_block(nn, filter_size=3, dilated=False)
        
        nn = keras.layers.AveragePooling1D(pool_size=4, 
                                           strides=4, 
                                           )(nn)
        nn = keras.layers.Dropout(0.3)(nn)
        #""
        # Fully-connected NN
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(256, activation=None, use_bias=False)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        # output layer
        outputs = keras.layers.Dense(self.num_class, activation='linear', use_bias=True)(nn)
        
        if self.classification:
            outputs = keras.layers.Activation('sigmoid')(outputs)

        return keras.Model(inputs=inputs, outputs=outputs)
"""


###############################################################


def load_rnacompete_data(file_path, ss_type='seq', normalization='log_norm', rbp_index=None, dataset_name=None):

    def prepare_data(train, ss_type=None):

        seq = train['inputs'][:,:,:4]

        if ss_type == 'pu':
            structure = train['inputs'][:,:,4:9]
            paired = np.expand_dims(structure[:,:,0], axis=2)
            unpaired = np.expand_dims(np.sum(structure[:,:,1:], axis=2), axis=2)
            seq = np.concatenate([seq, paired, unpaired], axis=2)

        elif ss_type == 'struct':
            structure = train['inputs'][:,:,4:9]
            paired = np.expand_dims(structure[:,:,0], axis=2)
            HIME = structure[:,:,1:]
            seq = np.concatenate([seq, paired, HIME], axis=2)

        train['inputs']  = seq
        return train

    def normalize_data(data, normalization):
        if normalization == 'clip_norm':
            # standard-normal transformation
            significance = 4
            std = np.std(data)
            index = np.where(data > std*significance)[0]
            data[index] = std*significance
            mu = np.mean(data)
            sigma = np.std(data)
            data_norm = (data-mu)/sigma
            params = [mu, sigma]

        elif normalization == 'log_norm':
            # log-standard-normal transformation
            MIN = np.min(data)
            data = np.log(data-MIN+1)
            mu = np.mean(data)
            sigma = np.std(data)
            data_norm = (data-mu)/sigma
            params = [MIN, mu, sigma]
        return data_norm, params

    # open dataset
    dataset = h5py.File(file_path, 'r')
    if not dataset_name:  
        # load data from RNAcompete 2013
        X_train = np.array(dataset['X_train']).astype(np.float32)
        Y_train = np.array(dataset['Y_train']).astype(np.float32)
        X_valid = np.array(dataset['X_valid']).astype(np.float32)
        Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
        X_test = np.array(dataset['X_test']).astype(np.float32)
        Y_test = np.array(dataset['Y_test']).astype(np.float32)

        # expand dims of targets
        if rbp_index is not None:
            Y_train = Y_train[:,rbp_index]
            Y_valid = Y_valid[:,rbp_index]
            Y_test = Y_test[:,rbp_index]
    else:
        #print(f"DEBUG {dataset_name=}")
        # necessary for RNAcompete 2009 dataset
        X_train = np.array(dataset['/'+dataset_name+'/X_train']).astype(np.float32)
        Y_train = np.array(dataset['/'+dataset_name+'/Y_train']).astype(np.float32)
        X_valid = np.array(dataset['/'+dataset_name+'/X_valid']).astype(np.float32)
        Y_valid = np.array(dataset['/'+dataset_name+'/Y_valid']).astype(np.float32)
        X_test = np.array(dataset['/'+dataset_name+'/X_test']).astype(np.float32)
        Y_test = np.array(dataset['/'+dataset_name+'/Y_test']).astype(np.float32)

    # expand dims of targets if needed
    if len(Y_train.shape) == 1:
        Y_train = np.expand_dims(Y_train, axis=1)
        Y_valid = np.expand_dims(Y_valid, axis=1)
        Y_test = np.expand_dims(Y_test, axis=1)

    #""
    # transpose to make (N, L, A)
    X_train = X_train.transpose([0, 2, 1])
    X_test = X_test.transpose([0, 2, 1])
    X_valid = X_valid.transpose([0, 2, 1])
    #""
    
    # filter NaN
    train_index = np.where(np.isnan(Y_train) == False)[0]
    valid_index = np.where(np.isnan(Y_valid) == False)[0]
    test_index = np.where(np.isnan(Y_test) == False)[0]
    Y_train = Y_train[train_index]
    Y_valid = Y_valid[valid_index]
    Y_test = Y_test[test_index]
    X_train = X_train[train_index]
    X_valid = X_valid[valid_index]
    X_test = X_test[test_index]

    # normalize intenensities
    Y_train, params_train = normalize_data(Y_train, normalization)
    Y_valid, params_valid = normalize_data(Y_valid, normalization)
    Y_test, params_test = normalize_data(Y_test, normalization)

    # dictionary for each dataset
    train = {'inputs': X_train, 'targets': Y_train}
    valid = {'inputs': X_valid, 'targets': Y_valid}
    test = {'inputs': X_test, 'targets': Y_test}

    # parse secondary structure profiles
    train = prepare_data(train, ss_type)
    valid = prepare_data(valid, ss_type)
    test = prepare_data(test, ss_type)

    return train, valid, test








#############################################################

if __name__=='__main__':

    # https://github.com/p-koo/residualbind/blob/master/train_rnacompete_2013.py

    normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
    ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
    #data_path = '../inputs/rnacompete2013.h5' #'../data/RNAcompete_2013/rnacompete2013.h5' # https://data.mendeley.com/datasets/m2yzh6ktzb/1
    data_path = '../inputs/rnacompete2009.h5'
    #results_path = '../../outputs_DAL/results_rnacompete2013' #helper.make_directory('../results', 'rnacompete_2013')
    results_path = '../../outputs_DAL/results_rnacompete2009'
    save_path = results_path+'/'+normalization+'_'+ss_type # helper.make_directory(results_path, normalization+'_'+ss_type)

    # loop over different RNA binding proteins
    pearsonr_scores = []
    #experiments = helper.get_experiment_names(data_path)
    #experiments=[i.decode('UTF-8') for i in np.array(h5py.File(data_path, 'r')['experiment'])] # 2013
    experiments = ['VTS1'] #'Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1'] # 2009
    print(f"{experiments=}")

    for rbp_index, experiment in enumerate(experiments):
        if 'RBFOX' in experiment: print(rbp_index,experiment)
        if 'VTS' in experiment: print(rbp_index,experiment)

    for rbp_index, experiment in enumerate(experiments):
        print('Analyzing: '+ experiment)

        # load rbp dataset
        #train, valid, test = load_rnacompete_data(data_path, ss_type=ss_type, normalization=normalization, rbp_index=rbp_index) #2013
        train, valid, test = load_rnacompete_data(data_path, ss_type=ss_type, normalization=normalization, dataset_name=experiment) #2009

        print(f"{train['inputs'].shape=} {train['targets'].shape=}")
        #exit()

        # load residualbind model
        input_shape = list(train['inputs'].shape)[1:]
        #print(f"{list(train['inputs'].shape)=}") # [37187, 39, 4]
        print(f"{input_shape=}") # [39, 4]
        num_class = 1
        weights_path = os.path.join(save_path, experiment + '_weights.hdf5')    
        resnet = ResidualBind_AC(input_shape, num_class, weights_path, with_residual=True)
        #resnet = ResidualBind_Orig(input_shape, num_class, weights_path)

        #proc_seqs=torch.tensor(np.transpose(np.array(valid['inputs']), (0, 2, 1)))[0]
        proc_seqs=torch.tensor(np.transpose(np.array(valid['inputs']), (0, 2, 1)))[0:10]
        pred_AC=resnet(proc_seqs)
        print(f"{pred_AC=}")

        exit() # AC: haven't adapted the script after this point.

        # fit model
        resnet.fit(train, valid, num_epochs=300, batch_size=100, patience=20, lr=0.001, lr_decay=0.3, decay_patience=7)
            
        # evaluate model
        metrics = resnet.test_model(test, batch_size=100, load_weights='best')
        print("  Test: "+str(np.mean(metrics)))

        pearsonr_scores.append(metrics)
    pearsonr_scores = np.array(pearsonr_scores)

    print('FINAL RESULTS: %.4f+/-%.4f'%(np.mean(pearsonr_scores), np.std(pearsonr_scores)))

    # save results to table
    file_path = os.path.join(results_path, normalization+'_'+ss_type+'_performance.tsv')
    with open(file_path, 'w') as f:
        f.write('%s\t%s\n'%('Experiment', 'Pearson score'))
        for experiment, score in zip(experiments, pearsonr_scores):
            f.write('%s\t%.4f\n'%(experiment, score))
