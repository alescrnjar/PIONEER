import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_initialization(conv_filters, initialization='kaiming_normal'):
    if initialization=='kaiming_normal': 
        nn.init.kaiming_normal_(conv_filters) #AC orig
    elif initialization=='xavier_normal': 
        nn.init.xavier_normal_(conv_filters) # https://pytorch.org/docs/stable/nn.init.html
    elif initialization=='kaiming_uniform': 
        nn.init.kaiming_uniform_(conv_filters)
    elif initialization=='xavier_uniform': 
        nn.init.xavier_uniform_(conv_filters)
    elif initialization=='normal_0.001': 
        scale=0.001 #Chandana's recommendation
        torch.nn.init.normal_(conv_filters, std=scale)
    elif initialization=='normal_0.005': 
        scale=0.005
        torch.nn.init.normal_(conv_filters, std=scale)
    return conv_filters

def apply_initialization_to_dense(layer, initialization='kaiming_normal'):
    if initialization=='kaiming_normal': 
        nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(layer.bias)
    elif initialization=='xavier_normal': 
        nn.init.xavier_normal_(layer.weight) # https://pytorch.org/docs/stable/nn.init.html
        nn.init.xavier_normal_(layer.bias) 
    elif initialization=='kaiming_uniform': 
        nn.init.kaiming_uniform_(layer.weight)
        nn.init.kaiming_uniform_(layer.bias)
    elif initialization=='xavier_uniform': 
        nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(layer.bias)
    elif initialization=='normal_0.001': 
        scale=0.001 #Chandana's recommendation
        torch.nn.init.normal_(layer.weight, std=scale)
        torch.nn.init.normal_(layer.bias, std=scale)
    elif initialization=='normal_0.005': 
        scale=0.005
        torch.nn.init.normal_(layer.weight, std=scale)
        torch.nn.init.normal_(layer.bias, std=scale)
    return layer

"""
def apply(x):
    x+=2
    return x

class Boh():
    def __init__(self):
        self.x=10
        self.x=apply(self.x)

boh=Boh()
print(boh.x)
"""

class DeepSTARR(nn.Module):
    # https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
    """DeepSTARR model from de Almeida et al., 2022; 
        see <https://www.nature.com/articles/s41588-022-01048-5>
    """
    def __init__(self, output_dim, d=256, 
                 initialization='kaiming_normal', #AC 
                 initialize_dense=False, #AC
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()
        
        if d != 256:
            print("NB: number of first-layer convolutional filters in original DeepSTARR model is 256; current number of first-layer convolutional filters is not set to 256")
        
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        self.AC_dropout = nn.Dropout(0.2) # AC
        
        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters
        
        assert (not (conv1_filters is None and not learn_conv1_filters)), "initial conv1_filters cannot be set to None while learn_conv1_filters is set to False"
        assert (not (conv2_filters is None and not learn_conv2_filters)), "initial conv2_filters cannot be set to None while learn_conv2_filters is set to False"
        assert (not (conv3_filters is None and not learn_conv3_filters)), "initial conv3_filters cannot be set to None while learn_conv3_filters is set to False"
        assert (not (conv4_filters is None and not learn_conv4_filters)), "initial conv4_filters cannot be set to None while learn_conv4_filters is set to False"

        # Layer 1 (convolutional), constituent parts
        if conv1_filters is not None:
            if learn_conv1_filters: # continue modifying existing conv1_filters through learning
                self.conv1_filters = nn.Parameter( torch.Tensor(conv1_filters) )
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            self.conv1_filters=apply_initialization(self.conv1_filters, initialization=initialization)

        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU() # name the first-layer activation function for hook purposes #QUIQUIURG is it ok that the first conv is the only one with a ReLU? In the model_zoo is the same though: https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
        self.maxpool1 = nn.MaxPool1d(2)
        
        # Layer 2 (convolutional), constituent parts
        if conv2_filters is not None:
            if learn_conv2_filters: # continue modifying existing conv2_filters through learning
                self.conv2_filters = nn.Parameter( torch.Tensor(conv2_filters) )
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            self.conv2_filters=apply_initialization(self.conv2_filters, initialization=initialization)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Layer 3 (convolutional), constituent parts
        if conv3_filters is not None:
            if learn_conv3_filters: # continue modifying existing conv3_filters through learning
                self.conv3_filters = nn.Parameter( torch.Tensor(conv3_filters) )
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            self.conv3_filters=apply_initialization(self.conv3_filters, initialization=initialization)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)
        
        # Layer 4 (convolutional), constituent parts
        if conv4_filters is not None:
            if learn_conv4_filters: # continue modifying existing conv4_filters through learning
                self.conv4_filters = nn.Parameter( torch.Tensor(conv4_filters) )
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            self.conv4_filters=apply_initialization(self.conv4_filters, initialization=initialization)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)
        
        # Layer 5 (fully connected), constituent parts
        #self.fc5 = nn.LazyLinear(256, bias=True) #greatold 20 nov 2024
        self.fc5 = nn.Linear(1800,256, bias=True) #AC: inserted to try and fix why bmdal does not work for DeepSTARR
        #torch.nn.init.kaiming_uniform_(self.fc5.weight)
        #torch.nn.init.normal_(self.fc5_weight, std=scale)
        if initialize_dense: self.fc5=apply_initialization_to_dense(self.fc5,initialization=initialization)
        self.batchnorm5 = nn.BatchNorm1d(256)

        # Layer 6 (fully connected), constituent parts
        #""
        self.fc6 = nn.Linear(256, 256, bias=True)
        #torch.nn.init.kaiming_uniform_(self.fc6.weight)
        #torch.nn.init.normal_(self.fc6_weight, std=scale)
        if initialize_dense: self.fc6=apply_initialization_to_dense(self.fc6,initialization=initialization)
        self.batchnorm6 = nn.BatchNorm1d(256)
        #""
        ##boh=nn.Parameter(torch.zeros(256, 256))
        ##torch.nn.init.kaiming_uniform_(boh)
        ##self.fc6 = nn.Linear(boh, bias=True)
        ##self.batchnorm6 = nn.BatchNorm1d(256)

        # Output layer (fully connected), constituent parts
        self.fc7 = nn.Linear(256, output_dim)
        #torch.nn.init.kaiming_uniform_(self.fc7.weight) # QUIQUIURG or .parameters??? 
        #torch.nn.init.normal_(self.fc7_weight, std=scale)
        if initialize_dense: self.fc7=apply_initialization_to_dense(self.fc7,initialization=initialization)
        
    def get_which_conv_layers_transferred(self):
        layers = []
        if self.init_conv1_filters is not None:
            layers.append(1)
        if self.init_conv2_filters is not None:
            layers.append(2)
        if self.init_conv3_filters is not None:
            layers.append(3)
        if self.init_conv4_filters is not None:
            layers.append(4)
        return layers
    
    def forward(self, x):
        # Layer 1
        #print(f"{self.conv1_filters.shape=}")
        #print(f"{self.batchnorm1=}")
        #print(f"{self.maxpool1=}")
        #print()
        #print(f"{self.conv2_filters.shape=}")
        #print(f"{self.batchnorm2=}")
        #print(f"{self.maxpool2=}")
        #
        #exit()
        """
        self.conv1_filters.shape=torch.Size([256, 4, 7])
        self.batchnorm1=BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1=MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2_filters.shape=torch.Size([60, 256, 3])
        self.batchnorm2=BatchNorm1d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2=MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        """
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)
        cnn = self.AC_dropout(cnn) #AC
        
        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)
        cnn = self.AC_dropout(cnn) #AC
        
        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)
        cnn = self.AC_dropout(cnn) #AC
        
        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool4(cnn)
        cnn = self.AC_dropout(cnn) #AC
        
        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Output layer
        y_pred = self.fc7(cnn) 
        
        return y_pred



