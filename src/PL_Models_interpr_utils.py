# https://github.com/alescrnjar/Interpreters4Motifs/blob/main/AminoPWM.py
# https://colab.research.google.com/drive/1SWDIKr2uIITAkUYJEvVXK8E6sR-X9v30

from PL_Models import *
import logomaker
from captum.attr import Saliency
from captum.attr import GradientShap
import sys
sys.path.append('../')
import outflag_2_nickname
import os
import torch.utils.data
import tqdm
import Experiments
import argparse
import glob

def apply_gradient_correction(grad):
    corrected_grad=grad-torch.mean(grad,axis=1,keepdims=True) #N,4,L
    return corrected_grad
def captum_sal(x_test,y_test,model,index=9):
    x = np.expand_dims(x_test[index,:,:], axis=0)
    y = np.expand_dims(y_test[index], axis=0)
    x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model.model.eval() #AC

    saliency = Saliency(model)
    grad = saliency.attribute(x_tensor) #, target=pred_label_idx)
    #grad = grad.data.cpu().numpy()
    grad = grad.data
    #print(f"{grad.shape=}")
    return grad


def captum_gradientshap(x_test,y_test,model,index=100,todevice=True, apply_corr=True, null_method='standard', device='cuda', num_background = 1000): #Amber: 1000 #20
    #print("------------- INTO GRADIENTSHAP")
    x = np.expand_dims(x_test[index,:,:], axis=0)
    y = np.expand_dims(y_test[index], axis=0)

    if todevice:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    else:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

    model.model.eval() #AC
    L=x_test.shape[2] #AC
    A=4 #AC

    null_index = np.random.randint(0,3, size=(num_background,L))
    x_null = np.zeros((num_background, A, L))
    for n in range(num_background):
        for l in range(L):
            x_null[n,null_index[n,l],l] = 1.0
    if todevice:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32).to(device)
    else:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32)

    gradient_shap = GradientShap(model)
    if todevice:
        grad = gradient_shap.attribute(x_tensor.to(device), # https://captum.ai/api/gradient_shap.html
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx #AC orig #AC: checked: with or without this doesnt make a difference
                              )
    else:
        grad = gradient_shap.attribute(x_tensor,
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx # AC orig
                              )

    #grad = grad.data.cpu().numpy()
    grad = grad.data

    if apply_corr:
        grad=apply_gradient_correction(grad)

    return grad

def captum_gradientshap_no_y(x_test,model,index=100,todevice=True, apply_corr=True, null_method='standard', device='cuda', num_background = 1000): #Amber: 1000 #20
    #print("------------- INTO GRADIENTSHAP")
    x = np.expand_dims(x_test[index,:,:], axis=0)

    if todevice:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    else:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    model.model.eval() #AC
    L=x_test.shape[2] #AC
    A=4 #AC

    #""
    #print("AC: random background")
    #num_background = 20
    null_index = np.random.randint(0,3, size=(num_background,L))
    x_null = np.zeros((num_background, A, L))
    for n in range(num_background):
        for l in range(L):
            x_null[n,null_index[n,l],l] = 1.0
    if todevice:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32).to(device)
    else:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32)

    #gradient_shap = GradientShap(model.to(device)) #goodold july 2 2024
    if todevice:
        gradient_shap = GradientShap(model.to(device))
    else:
        gradient_shap = GradientShap(model)
    if todevice:
        grad = gradient_shap.attribute(x_tensor.to(device), # https://captum.ai/api/gradient_shap.html
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx #AC orig #AC: checked: with or without this doesnt make a difference
                              )
    else:
        grad = gradient_shap.attribute(x_tensor,
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx # AC orig
                              )

    #grad = grad.data.cpu().numpy()
    grad = grad.data

    if apply_corr:
        grad=apply_gradient_correction(grad)

    return grad

def captum_gradientshap_no_y_no_index(x_test,model,todevice=True, apply_corr=True, null_method='standard', device='cuda', num_background = 1000): #Amber: 1000 #20
    #print("------------- INTO GRADIENTSHAP")
    x = x_test # np.expand_dims(x_test[index,:,:], axis=0)

    if todevice:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    else:
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    model.model.eval() #AC
    L=x_test.shape[2] #AC
    A=4 #AC

    #""
    #print("AC: random background")
    #num_background = 20
    null_index = np.random.randint(0,3, size=(num_background,L))
    x_null = np.zeros((num_background, A, L))
    for n in range(num_background):
        for l in range(L):
            x_null[n,null_index[n,l],l] = 1.0
    if todevice:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32).to(device)
    else:
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32)
    #gradient_shap = GradientShap(model.to(device)) #goodold july 2 2024
    if todevice:
        gradient_shap = GradientShap(model.to(device))
    else:
        gradient_shap = GradientShap(model)
    if todevice:
        grad = gradient_shap.attribute(x_tensor.to(device), # https://captum.ai/api/gradient_shap.html
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx #AC orig #AC: checked: with or without this doesnt make a difference
                              )
    else:
        grad = gradient_shap.attribute(x_tensor,
                              n_samples=100,
                              stdevs=0.1,
                              baselines=x_null_tensor,
                              #target=pred_label_idx # AC orig
                              )

    #grad = grad.data.cpu().numpy()
    grad = grad.data

    if apply_corr:
        grad=apply_gradient_correction(grad)

    return grad
