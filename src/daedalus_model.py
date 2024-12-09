# https://colab.research.google.com/drive/1N5oI2eO8NnVXJ6UxehyIRky4BikH1gNd?usp=sharing

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math

##############################################################

"""
class MultiHeadAttention2(nn.Module):
    def __init__(self, d_model, num_heads, embedding_size=None):
        super(MultiHeadAttention2, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size is None else embedding_size
        assert d_model % self.num_heads == 0 and d_model % 6 == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False) # The query determines which values to focus on; we can say that the query ‘attends’ to the values. https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.r_k_layer = nn.Linear(embedding_size, d_model, bias=False)
        self.r_w = nn.Parameter(torch.randn(1, self.num_heads, 1, self.depth) * 0.5, requires_grad=True)
        self.r_r = nn.Parameter(torch.randn(1, self.num_heads, 1, self.depth) * 0.5, requires_grad=True)

        self.dense = nn.Linear(d_model, d_model)

    def forward(self, v, k, q):
        batch_size, seq_len, _ = q.size()

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        q = q / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        pos = torch.arange(-seq_len + 1, seq_len, dtype=torch.float32).unsqueeze(0)
        feature_size = self.embedding_size // 6

        seq_length = seq_len.float()
        exp1 = self.f_exponential(torch.abs(pos), feature_size, seq_length=seq_length)
        exp2 = exp1 * torch.sign(pos).unsqueeze(-1)
        cm1 = self.f_central_mask(torch.abs(pos), feature_size, seq_length=seq_length)
        cm2 = cm1 * torch.sign(pos).unsqueeze(-1)
        gam1 = self.f_gamma(torch.abs(pos), feature_size, seq_length=seq_length)
        gam2 = gam1 * torch.sign(pos).unsqueeze(-1)

        positional_encodings = torch.cat([exp1, exp2, cm1, cm2, gam1, gam2], dim=-1)
        positional_encodings = F.dropout(positional_encodings, p=0.1, training=self.training)

        r_k = self.r_k_layer(positional_encodings)
        r_k = r_k.view(r_k.size(0), r_k.size(1), self.num_heads, self.depth).permute(0, 2, 1, 3)

        content_logits = torch.matmul(q + self.r_w, k.transpose(-2, -1))
        relative_logits = torch.matmul(q + self.r_r, r_k.transpose(-2, -1))
        relative_logits = self.relative_shift(relative_logits)

        logits = content_logits + relative_logits
        attention_map = F.softmax(logits, dim=-1)

        attended_values = torch.matmul(attention_map, v)
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.dense(attended_values)
        return output, attention_map

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def f_exponential(self, positions, feature_size, seq_length=None, min_half_life=3.0):
        if seq_length is None:
            seq_length = torch.max(torch.abs(positions)) + 1.0
        max_range = torch.log(seq_length) / torch.log(2.0)
        half_life = 2.0 ** torch.linspace(min_half_life, max_range, feature_size)
        half_life = half_life.view(1, *([1] * positions.dim()), -1)
        positions = torch.abs(positions)
        outputs = torch.exp(-torch.log(2.0) / half_life * positions.unsqueeze(-1))
        return outputs

    def f_central_mask(self, positions, feature_size, seq_length=None):
        center_widths = 2.0 ** torch.arange(1, feature_size + 1).float() - 1
        center_widths = center_widths.view(1, *([1] * positions.dim()), -1)
        outputs = (center_widths > torch.abs(positions).unsqueeze(-1)).float()
        return outputs

    def f_gamma(self, positions, feature_size, seq_length=None):
        if seq_length is None:
            seq_length = torch.max(torch.abs(positions)) + 1
        stdv = seq_length / (2.0 * feature_size)
        start_mean = seq_length / feature_size
        mean = torch.linspace(start_mean, seq_length, steps=feature_size).view(1, *([1] * positions.dim()), -1)
        concentration = (mean / stdv) ** 2
        rate = mean / stdv ** 2

        def gamma_pdf(x, conc, rt):
            log_unnormalized_prob = conc.log() * x - rate * x
            log_normalization = (conc.lgamma() - concentration * rate.log())
            return (log_unnormalized_prob - log_normalization).exp()

        probabilities = gamma_pdf(torch.abs(positions).float().unsqueeze(-1), concentration, rate)
        outputs = probabilities / probabilities.max()
        return outputs

    def relative_shift(self, x):
        x = F.pad(x, (0, 0, 1, 0))
        batch_size, num_heads, t1, t2 = x.size()
        x = x.view(-1, num_heads, t2, t1)
        x = x[:, :, 1:, :]
        x = x.view(-1, num_heads, t1, t2 - 1)
        x = x[:, :, :, :((t2 + 1) // 2)]
        return x
"""
        

############################################################################

"""
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_dim, in_dim)
        self.W_k = nn.Linear(in_dim, in_dim)
        self.W_v = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = self.softmax(attention_scores)
        attended_values = torch.bmm(attention_weights, value)
        
        return attended_values

class In_House_CNN_Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(In_House_CNN_Attention, self).__init__()
        
        self.attention = SelfAttention(input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.2)
        
        # Add more layers with attention if needed
        self.fc4 = nn.Linear(256, 256)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(256, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        attended = self.attention(x)
        x = self.conv1(attended)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        # Add more layers with attention if needed
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.activation4(x)
        x = self.dropout4(x)
        
        logits = self.fc5(x) 
        y_pred = self.output_activation(logits)
        
        return y_pred
"""

"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        # Create the positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #0::2 starts from 0 and progresses with a stride of 2: every even
        pe[:, 1::2] = torch.cos(position * div_term) #1::2 starts from 0 and progresses with a stride of 2: every odd
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) # "" If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers. Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them. # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723 ""

    def forward(self, x):
        print(f"{x.size()=}")
        print(f"{self.pe[:,x.size(1)].shape=}")
        x_ped = x + self.pe[:, :x.size(1)]
        print(f"PE: {x=} {x_ped=}")
        return self.dropout(x_ped)
"""

# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        # Create the positional encodings
        #pe = torch.zeros(1, max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) 
        division_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        #print(f"{position.shape=}")
        #print(f"{division_term.shape=}")
        #exit()
        #position = position.expand(-1, d_model // 2)
        #division_term = division_term.expand(max_seq_len, -1)
        #print(f"{position.shape=}")
        #print(f"{division_term.shape=}")

        #pe[0, :, 0::2] = torch.sin(position * division_term) #0::2 starts from 0 and progresses with a stride of 2: every even
        #pe[0, :, 1::2] = torch.cos(position * division_term) #1::2 starts from 0 and progresses with a stride of 2: every odd
        pe[:, 0::2] = torch.sin(position * division_term) #0::2 starts from 0 and progresses with a stride of 2: every even
        pe[:, 1::2] = torch.cos(position * division_term) #1::2 starts from 0 and progresses with a stride of 2: every odd
        #sin=torch.sin(position * division_term)
        #cos=torch.cos(position * division_term)
        #print(f"{sin.shape=} {cos.shape=} {pe.shape=}")
        #exit()
        #print(f"{pe.shape=}") # shape:: max_seq_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
    #def forward(self, x_embedded):
        #print(f"{x_embedded.shape=} {self.pe.shape=} {self.pe[:, :x.size(1)].shape=}")
        #exit()
        #x = x + self.pe
        #print(f"{x.size()=}")
        #print(f"{self.pe[:,x.size(1)].shape=}")
        #x_ped = x + self.pe[:, :x.size(1)]
        #x = x + self.pe[:, :x.size(2)].transpose(0, 2)
        #print(f"PE: {x=} {x_ped=}")
        #print(f"{x_embedded.shape=} {self.pe[:x_embedded.size(0), :].shape=}")
        #return self.dropout(x_embedded + self.pe[:x_embedded.size(0), :])
        #return self.dropout(x)
        #print(f"{x.shape=} {self.pe.shape=} {self.pe[:,:x.size(2)].shape=}")
        #x = x + self.pe[:, :x.size(2)].transpose(0, 2).unsqueeze(0)
        x = x + self.pe.transpose(0, 1) ##QUIQUIURG or permute???
        ##x = x + self.pe.permute(0, 1)
        return x
        

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim=in_dim
        self.W_q = nn.Linear(in_dim, in_dim)
        self.W_k = nn.Linear(in_dim, in_dim)
        self.W_v = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_weights=None

    def forward(self, x):
        ##x=x.transpose(1,2)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        ##attention_scores = torch.bmm(query, key.transpose(1, 2)) #batch matrix matrix product
        attention_scores = torch.matmul(query.transpose(1,2), key) / math.sqrt(self.in_dim)
        attention_weights = self.softmax(attention_scores)
        self.attention_weights=attention_weights
        #attended_values = torch.bmm(attention_weights, value)
        attended_values = torch.bmm(attention_weights, value.transpose(1,2))
        
        attended_values=attended_values.transpose(1,2) #https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py
        return attended_values
    
    def get_attention_weights(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        
        ##attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = torch.matmul(query.transpose(1,2), key) / math.sqrt(self.in_dim)
        attention_weights = self.softmax(attention_scores)
        return attention_weights
    
class MultiHeadSelfAttention(nn.Module):
    """
    ChatGPT: This modified MultiHeadSelfAttention module allows you to create multiple heads, each with its own set of weight matrices for queries, keys, and values. It processes the input sequence independently for each head, and the results are concatenated to capture different patterns or dependencies in the input data.
    """
    def __init__(self, in_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        # Create learnable weight matrices for queries, keys, and values for each head
        self.W_q = nn.Parameter(torch.randn(num_heads, in_dim, self.head_dim))
        self.W_k = nn.Parameter(torch.randn(num_heads, in_dim, self.head_dim))
        self.W_v = nn.Parameter(torch.randn(num_heads, in_dim, self.head_dim))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        head_outputs = []
        for head in range(self.num_heads):
            query = torch.matmul(x, self.W_q[head])
            key = torch.matmul(x, self.W_k[head])
            value = torch.matmul(x, self.W_v[head])
            
            #attention_scores = torch.bmm(query, key.transpose(1, 2))
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = self.softmax(attention_scores)
            attended_values = torch.bmm(attention_weights, value)
            
            head_outputs.append(attended_values)
        
        # Concatenate the outputs from all heads
        concatenated = torch.cat(head_outputs, dim=-1)
        
        return concatenated


class In_House_CNN_Attention_Positional(nn.Module):
    def __init__(self, seq_len, n_firstlayer_filters, #emb_dim, 
                 output_dim, final_activation=None):
        super(In_House_CNN_Attention_Positional, self).__init__()
        #self.embedding = nn.Embedding(n_firstlayer_filters, emb_dim)
        ##self.positional_encoding = PositionalEncoding(n_firstlayer_filters, seq_len)
        self.positional_encoding = PositionalEncoding(4, seq_len)
        #self.positional_encoding = PositionalEncoding(emb_dim, seq_len)
        #self.attention = SelfAttention(n_firstlayer_filters)
        self.attention = SelfAttention(seq_len)
        #self.conv1 = nn.Conv1d(n_firstlayer_filters, 4, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(4, n_firstlayer_filters, kernel_size=7, padding=3)
        # DSRR: self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
        # (N,C_in,L) -> (N,C_out, L_out)
        self.batchnorm1 = nn.BatchNorm1d(n_firstlayer_filters)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.2)
        
        #print(f"{self.conv1(torch.rand((1,4,249))).size()}= {self.conv1(torch.rand((1,4,251))).size()}=")
        #print(f"{self.maxpool1(self.conv1(torch.rand((1,4,seq_len)))).size()=} {self.maxpool1(self.conv1(torch.rand((1,4,78)))).size()=}")

        maxpool_outsize=self.maxpool1(self.conv1(torch.rand((1,4,seq_len)))).size()[-1] #torch.Size([1, 256, 62])

        """
        # Add more layers with attention if needed
        #self.fc4 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(maxpool_outsize, 256)
        
        #self.fc4=nn.LazyLinear(256)
        
        #self.flatten=nn.Flatten()
        #self.fc4 = nn.Linear(maxpool_outsize*256, 256)
        
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2) # orig: 0.5

        self.fc5 = nn.Linear(256, output_dim)
        self.output_activation = nn.Sigmoid()
        """
        # Add more layers with attention if needed
        #self.fc4 = nn.Linear(256, 256)
        #self.fc4 = nn.Linear(maxpool_outsize, 256)
        
        #self.fc4=nn.LazyLinear(256)
        
        self.flatten=nn.Flatten()
        self.fc4 = nn.Linear(maxpool_outsize*256, 256)
        
        self.batchnorm4 = nn.BatchNorm1d(256) #to be applied after a flatten, requires to be fed at least 2 sequences(???)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2) # orig: 0.5

        self.fc5 = nn.Linear(256, output_dim)
        if final_activation!=None: 
            #self.output_activation = nn.Sigmoid()
            self.output_activation = eval('nn.'+str(final_activation)+'()')
        else:
            self.output_activation = None

    def forward(self, x):
        out=x
        out = self.positional_encoding(out)  # Add positional encoding
        #exit()
        attended = self.attention(out)
        #print(f"{attended.shape=}")
        #out = self.conv1(attended)
        out = self.conv1(out)
        #print(f"{out.shape=}")
        out = self.batchnorm1(out)
        #print(f"{out.shape=}")
        out = self.activation1(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)
        
        #print("---")
        # Add more layers with attention if needed
        #print(f"{out.shape=}")
        out=self.flatten(out)
        #print(f"{out.shape=}")
        out = self.fc4(out)
        #print(f"0 {out.shape=}")
        out = self.batchnorm4(out)
        out = self.activation4(out)
        out = self.dropout4(out)
        
        #print(f"{out.shape=}")
        logits = self.fc5(out)
        #print(f"{out.shape=}") 
        if self.output_activation!=None:  
            y_pred = self.output_activation(logits)
        else:
            y_pred=logits
        
        return y_pred

        

class ConvBlock(nn.Module):
    def __init__(self, input_dim, out_channels, kernel_size, stride=1, dilation=1, padding='same', initialization='kaiming_uniform', wanted_BatchNorm=True, wanted_relu=True, nMaxP=2, dropout_p=0.1):
        super(ConvBlock, self).__init__()
        #conv_filters = torch.nn.Parameter(torch.zeros(a, b, c))
        self.conv_filters = nn.Conv1d(in_channels=input_dim, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        #""
        if initialization=='kaiming_uniform':
            ##nn.init.kaiming_uniform_(self.conv_filters._parameters)
            nn.init.kaiming_uniform_(self.conv_filters.weight)
            #nn.init.kaiming_uniform_(self.conv_filters.bias) #ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        elif initialization=='kaiming_normal':
            ##nn.init.kaiming_normal_(self.conv_filters._parameters) #AC orig
            nn.init.kaiming_normal_(self.conv_filters.weight) #AC orig
            #nn.init.kaiming_normal_(self.conv_filters.bias) #AC orig
        elif initialization=='xavier_uniform':
            ##nn.init.xavier_uniform_(self.conv_filters._parameters)
            nn.init.xavier_uniform_(self.conv_filters.weight)
            #nn.init.xavier_uniform_(self.conv_filters.bias)
        elif initialization=='xavier_normal':
            ##nn.init.xavier_normal_(self.conv_filters._parameters) # https://pytorch.org/docs/stable/nn.init.html
            nn.init.xavier_normal_(self.conv_filters.weight) # https://pytorch.org/docs/stable/nn.init.html
            #nn.init.xavier_normal_(self.conv_filters.bias) # https://pytorch.org/docs/stable/nn.init.html
        elif initialization=='normal':
            scale=0.001 #0.005 0.001
            nn.init.normal_(self.conv_filters, std=scale)
        else:
            print("Wrong initialization")
            exit()
        #""
        if wanted_BatchNorm: 
            self.batchnorm = nn.BatchNorm1d(out_channels) #turn off to see how the training gets poorer, as experiment
        else:
            self.batchnorm = None
        if wanted_relu: 
            self.activation = nn.ReLU() # name the first-layer activation function for hook purposes #QUIQUIURG: maybe we only want it in the FIRST conv, like in DeepSTARR and basset: https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
        else:
            self.activation = None
        self.maxpool = nn.MaxPool1d(nMaxP)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out=x
        out=self.conv_filters(out)
        if self.batchnorm != None: out = self.batchnorm(out)
        if self.activation != None: out = self.activation(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        pass
    def forward(self, x):
        pass

class GCNConvBlock(nn.Module): #vedi GCNConv_test.py
    def __init__(self):
        super(GCNConvBlock, self).__init__()
        pass
    def forward():
        pass


class Daedalus(nn.Module):
    def __init__(self, 
                 wanted_initial_attention=True,
                 #nConvBlocks=2, nDense=2,
                 #first_input_dim=256, first_output_dim=64, 
                 convs=[[4,7,1],[256,7,1],[64,7,1],[32,0,0]],
                 linears=[16,8,4],
                 seq_len=200, 
                 initialization='kaiming_uniform', 
                 wanted_BatchNorm=True, 
                 wanted_relu=True,
                 out_activation=None 
                 ):
        super(Daedalus, self).__init__()
        self.wanted_initial_attetion=wanted_initial_attention
        if wanted_initial_attention:
            self.pe=PositionalEncoding(4, seq_len)
            self.attention=SelfAttention(seq_len)

        ConvBlocks=[]
        """
        input_dim=first_input_dim
        out_channels=first_output_dim
        for i_ConvBlock in range(nConvBlocks):
            convblock=ConvBlock(input_dim, out_channels, kernel_size, stride=1, dilation=1, padding='same', initialization='kaiming', wanted_BatchNorm=wanted_BatchNorm, nMaxP=2, dropout=0.1)
            self.ResBlocks.append(convblock)
            outsize=convblock(torch.rand((1,4,seq_len))).size()[-1] 
            input_dim=
        """
        for i_ConvBlock in range(len(convs)-1):
            #print(f"---{i_ConvBlock=}")
            input_dim=convs[i_ConvBlock][0]
            dummy_seq=torch.rand((1,input_dim,seq_len))
            kernel_size=convs[i_ConvBlock][1]
            dilation=convs[i_ConvBlock][2]
            out_channels=convs[i_ConvBlock+1][0]
            if i_ConvBlock==0: 
                wanted_relu=True
            else:
                wanted_relu=False # Like DeepSTARR and Basset, only in the first conv: https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
            convblock=ConvBlock(input_dim=input_dim, out_channels=out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding='same', initialization=initialization, wanted_BatchNorm=wanted_BatchNorm, wanted_relu=wanted_relu, nMaxP=2, dropout_p=0.1)
            #print(f"{convblock=}")
            #self.ConvBlocks.append(convblock)
            ConvBlocks.append(convblock)
            outsize=convblock(dummy_seq).size()[-1] 
            #print(f"{convblock(dummy_seq).size()=}")
        ##self.block1=convblock
        self.ConvBlocks=nn.ModuleList(ConvBlocks) # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463

        if len(convs)>0: # and len(linears)>0:
            self.flatten=nn.Flatten()

        #print(f"{convblock(dummy_seq).size()=}")
        #flattened_dim=convblock(dummy_seq).size()[-1]*convblock(dummy_seq).size()[-2] 
        flattened_dim=convblock(dummy_seq).size()[-2] 

        DenseLayers=[]
        #inp_dim=flattened_dim
        for i_Linear in range(len(linears)):
            out_dim=linears[i_Linear]
            #dummy_seq=torch.rand((1,inp_dim,seq_len))
            if i_Linear==0:
                denselayer=nn.LazyLinear(out_dim)
            else:
                denselayer=nn.Linear(inp_dim,out_dim)
            DenseLayers.append(denselayer)
            inp_dim=out_dim
        self.DenseLayers=nn.ModuleList(DenseLayers)

        if out_activation!=None: 
            #self.out_activation = nn.Sigmoid()
            self.out_activation = eval('nn.'+str(out_activation)+'()')
        else:
            self.out_activation = None
    
    
    def forward(self, x):
        out=x
        
        if self.wanted_initial_attetion:
            out=self.pe(out)
            out=self.attention(out)
        
        for i_ConvBlock in range(len(self.ConvBlocks)):
            out=self.ConvBlocks[i_ConvBlock](out)
            #print(f"Conv: {out.shape=}")

        if len(self.ConvBlocks)>0: # and len(self.DenseLayers)>0:
            out=self.flatten(out)

        #print(f"Flattened: {out.shape=}")

        for i_Linear in range(len(self.DenseLayers)):
            out=self.DenseLayers[i_Linear](out)
            #print(f"Dense: {out.shape=}")

        if self.out_activation!=None:
            y_pred=self.out_activation(out)
        else:
            y_pred=out

        return y_pred


        
if __name__=='__main__':
    #x=torch.rand((1,4,11)) #goodold
    #x=torch.rand((2,4,11))
    x=torch.rand((1,4,249))
    x=F.gumbel_softmax(x, hard=True, dim=1)
    #In House: self.X_train = torch.tensor(np.transpose(data[key_with_low(list(data.keys()),'x_train')][:, :4, :], (0, 1, 2)), dtype=torch.float32) #CNNAMBER
    #DSRR: self.X_train = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)

    ##x=x.transpose(2,1)
    print(f"{x[0,:,:10]=}")
    ##print(f"{x.shape=}")
    #emb=nn.Embedding(4, 19)
    #emb=nn.Embedding(11, 19)
    emb=nn.Embedding(num_embeddings=2, embedding_dim=3)
    #x_emb=emb(torch.tensor(x, dtype=torch.int))
    #print(f"{x_emb=}")
    #print(f"{x_emb.shape=}")
    #exit()

    """
    pos_enc=PositionalEncoding(d_model=4, max_seq_len=x.shape[-1])
    #### self.positional_encoding = PositionalEncoding(n_firstlayer_filters, seq_len)
    x_pe=pos_enc(x)
    ##x_pe=pos_enc(x_emb)
    print(f"{x_pe=} {x_pe.shape=}")

    att=SelfAttention(in_dim=11)
    #x_att=att(x)
    x_att=att(x_pe)
    print(f"{x_att=} {x_att.shape=}")
    exit()
    """

    """
    model=In_House_CNN_Attention_Positional(seq_len=x.shape[2], n_firstlayer_filters=256, output_dim=1, #emb_dim=4, 
                                            final_activation=None)
    model.eval() #necessary to have batchnorm post flatten work for single batch
    pred=model(x)
    #print(f"{pred.shape=}")
    print(f"{pred=}")
    """

    model=Daedalus(
                   wanted_initial_attention=True,
                   convs=[[4,7,1],[256,7,1],[64,7,1],[32,0,0]],
                   linears=[16,8,4,1],
                   seq_len=x.shape[2], 
                   initialization='kaiming_uniform', 
                   wanted_BatchNorm=True,
                   wanted_relu=True,
                   out_activation=None
                   )
    print("Results of ray tune (citra): results={'wanted_initial_attention': True, 'convs': [[4, 7, 1], [200, 7, 1], [100, 7, 1], [50, 0, 0]], 'linears': [20, 10, 1], 'initialization': 'kaiming_uniform', 'wanted_BatchNorm': True}")
    print(f"\n{model.parameters=}\n")
    model.eval() #necessary to have batchnorm post flatten work for single batch
    pred=model(x)
    print(f"{pred=}")


    print("\nTODOS")
    print("V 1) get prediction right with In_House w/o pos encoding and attention")
    print("V 2) intro pos enc and attention")
    print("V 3) port everything to Daedalus in order to have flexible architecture, to find with raytune")
    print("4) consider introducing a GConvBlock where GCNConv is used instead of conv, everything else is identical")
    print("INTRO SQRT FACTOR FOR SELF ATTENTION. SAPERE A COSA SERVE")
    print("5) Use wandb")
    print("conv before attention instead of after: like chandana (se non sbaglio la fonte di questa idea e questa)")
    print("6) Introduce ResBlocks")

    print()
    print("SCRIPT END")