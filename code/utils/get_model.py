from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import nn
import sys
import numpy as np
import json
import torch

from models.resnet import ResNet,assign_blocks
from models.branched_model import BranchedNet, VariableWidthBranchedNet
from models.weight_sharing_resnet import VariableWidthResNet

def create_model(model_hyperparameters):

    '''
    Function to create model given a hyperparameters dictionary

    Inputs
    -------
    model_hyperparameters: a dictionary containing
        model_name: the name of the model type to be loaded
        depth: the depth of the model
        width: the width multiplier on the model
        in_channels: the input channels to the model
        n_classes: the number of classes for classification

    Outputs:
    --------
    model: a pytorch model class
    '''

    #getting parameters from dictionary
    name = model_hyperparameters['backbone'].lower()
    depth = model_hyperparameters['depth']
    width = model_hyperparameters['width']
    in_channels = model_hyperparameters['in_channels']
    n_classes = model_hyperparameters['n_classes']
    res = model_hyperparameters['res']


    #define model using them
    if name == 'resnet': 
        #resnet needs to assign the blocks properly depending on the depth of the model
        blocks,block_type = assign_blocks(depth,basic_block = True)
        model = ResNet(in_channels, n_classes, blocks_sizes=[64, 128, 256, 512], block=block_type, depths=blocks, width_multiplier = width, resolution = res)
    
        #define model using them
    if name == 'slimmable_resnet': 
        #resnet needs to assign the blocks properly depending on the depth of the model
        #blocks_sizes needs to be defined on each call, otherwise it is overwritten on next call
        blocks,block_type = assign_blocks(depth,basic_block = True)
        model = VariableWidthResNet(in_channels, n_classes, block=block_type, depths=blocks, width=width, resolution = res)

    return model

def attach_branches(model_hyperparameters,model,n_branches):
    if 'slimmable' in model_hyperparameters['backbone']:
        return VariableWidthBranchedNet(model, n_branches)
    else:
        return BranchedNet(model, n_branches)


def load_pretrained(exp_name,model_params,data_props,device,weights_file=None,**kwargs):
    n_branches = model_params['n_branches']
    res = data_props['res']

    if 'res' not in model_params.keys():
        model_params['res'] = res

    n_channels = data_props['n_channels']
    input_shape = ((n_channels,res,res))

    if weights_file:
        weight_name = weights_file
    else:
        weight_name = 'weights_best.pth'

    if 'slimmable' in model_params['backbone']:
        weight_name = 'weights_final.pth'

        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches) 
        model.load_state_dict(torch.load(exp_name + "/" + weight_name,map_location=device))
        model.set_inference_mode(kwargs['width_mode'])

    else:
        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches)
        model.load_state_dict(torch.load(exp_name + "/" + weight_name, map_location=device))

    print('Loaded Weights at:',exp_name + "/" + weight_name)
    return(model)

def load_to_finetune(weights_directory,model_params,data_props,device,**kwargs):
    n_branches = model_params['n_branches']
    res = data_props['res']

    if 'res' not in model_params.keys():
        model_params['res'] = res

    n_channels = data_props['n_channels']
    input_shape = ((n_channels,res,res))

    if 'slimmable' in model_params['backbone']:

        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches) 
        model.load_state_dict(torch.load(weights_directory,map_location=device))
        model.set_inference_mode(kwargs['width_mode'])

    else:
        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches)
        model.load_state_dict(torch.load(weights_directory,map_location=device))

    print('Loaded Weights')
    return(model)