import os
import sys

import tqdm

if os.uname()[1]=='auros': 
    sys.path.append('/home/alessandro/Documents/GitHub_Local/bmdal_reg_custom/')
elif os.uname()[1]=='amethyst': 
    sys.path.append('/home/acrnjar/Desktop/TEMP/GitHub_Local/bmdal_reg_custom/')
else:
    sys.path.append('/grid/koo/home/crnjar/bmdal_reg_custom/')

from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch
from bmdal_reg.bmdal.algorithms import BatchSelectorImpl, MaxDetSelectionMethod #AC
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def batch_selection_method(
    x_train, x_pool, n_to_make, models, y_train,
    selection_method='maxdet',device='cuda',external_batch_size=100,
    base_kernel='grad', kernel_transforms=[('rp', [512])],
    sel_with_train=False,
    ):

    precomp_batch_size=external_batch_size #QUIQUIURG NOT USED!!!
    nn_batch_size=external_batch_size

    train_data = TensorFeatureData(torch.tensor(x_train))
    pool_data = TensorFeatureData(torch.tensor(x_pool))


    new_idxs, results_dict = select_batch(batch_size=n_to_make, models=models, 
                            data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                            selection_method=selection_method, sel_with_train=sel_with_train,
                            #base_kernel='grad', kernel_transforms=[('rp', [512])]) # return batch_idxs, results_dict : def select within class BatchSelectorImpl in algorithms.py
                            base_kernel=base_kernel, kernel_transforms=kernel_transforms) # return batch_idxs, results_dict : def select within class BatchSelectorImpl in algorithms.py
    return new_idxs
