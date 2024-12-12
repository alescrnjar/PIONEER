import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import os

class per_factor(nn.Module):
    def __init__(self, num_filters, filter_size, f):
        super(per_factor, self).__init__()
        self.activation=nn.ReLU()
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
    def __init__(self, num_filters, filter_size, activation='ReLU', dilated=False):
        super(residual_block, self).__init__()
        if dilated:
            factor = [2, 4, 8]
        else:
            factor = [1]
        self.conv = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, bias=False,padding='same',dilation=1)
        self.batchnorm = nn.BatchNorm1d(num_filters)

        per_factor_s=[]
        for i_f,f in enumerate(factor):
            per_factor_s.append(per_factor(num_filters, filter_size, f))
        self.per_factor_s=nn.ModuleList(per_factor_s) # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        
        if activation=='ReLU': # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
            self.final_activ = nn.ReLU() # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
        elif activation=='softplus_beta1': # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!
            self.final_activ = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20) # standardrelu1 TRUESOFT+?: there was a missing final activation here!!!

    def forward(self, x):
        out=x
        out=self.conv(out)
        out=self.batchnorm(out)
        for i_f in range(len(self.per_factor_s)):
            out=self.per_factor_s[i_f](out)
        out=out+x # AC: ResidualBind has a single (credo) skip connection
        out=self.final_activ(out) # standardrelu1 TRUESOFT+?
        return out

class ResidualBind(nn.Module):
    def __init__(self, input_shape=(41,4), num_class=1, classification=False, with_residual=True, activation='ReLU', wanted_initial_attention=False, wanted_hook=False):
        super(ResidualBind, self).__init__()
        self.input_shape = input_shape
        self.num_class = num_class
        self.classification = classification

        self.wanted_initial_attetion=wanted_initial_attention
        if wanted_initial_attention:
            self.pe=PositionalEncoding(4, input_shape[0])
            self.attention=SelfAttention(input_shape[0])
        
        # layer 1
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=96, kernel_size=11, stride=1, bias=False, padding='same') # padding=5)
        self.bn = nn.BatchNorm1d(96)
        if activation=='ReLU':
            self.activ = nn.ReLU()
        elif activation=='softplus_beta1':
            self.activ = torch.nn.Softplus(beta=int(activation.replace('softplus_beta','')), threshold=20)
        self.drop1 = nn.Dropout(0.1)

        # dilated residual block
        self.resblock=None
        num_filters = 96 #input_layer.shape.as_list()[-1]  
        if with_residual: self.resblock = residual_block(num_filters=num_filters, filter_size=3, dilated=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=10)
        self.drop2 = nn.Dropout(0.2)

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

        out=self.conv1(out)
        out=self.bn(out)
        layer1=self.activ(out)
        if self.wanted_hook: self.registered_hook = layer1.register_hook(lambda grad: setattr(self, 'first_layer_grad', grad))
        out=layer1        
        #
        out=self.drop1(out)

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
    

    def initial_attention_output(self, x):
        out=x
        if self.wanted_initial_attetion:
            out=self.pe(out)
            out=self.attention(out)
        return out







#############################################################################################



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





