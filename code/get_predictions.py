import os

import torch
from torch import nn
from torch.distributions import Categorical

import numpy as np
import argparse
from tqdm import tqdm
import json
import re

from utils.get_model import create_model,attach_branches
from utils.get_data import get_dataset
from utils.utils import get_branch_power_usage,slimmable_dict,dirichlet_probability, get_output_with_exits
from utils.analysis_utils import create_decision_module
    

def main(args):
    model_param_file = open(args.exp_name + '/model_params.json')
    model_params = json.load(model_param_file)
    n_branches = model_params['n_branches']

    train_loader,val_loader,test_loader,data_props = get_dataset(args.data,args.batch_size,shuffling=False)

    n_values_train = data_props['n_values_train']
    n_values_val = data_props['n_values_val']
    n_values_test = data_props['n_values_test']

    n_classes = data_props['classes']
    res = data_props['res']

    if 'res' not in model_params.keys():
        model_params['res'] = res

    n_channels = data_props['n_channels']
    input_shape = ((n_channels,res,res))

    if 'slimmable' in model_params['backbone']:

        power_params = dict(model_params)
        power_params['width'] = power_params['width']*slimmable_dict[args.width_mode]
        power_params['backbone'] = power_params['backbone'].strip('slimmable_')
        power_backbone = create_model(power_params)
        power_model = attach_branches(power_params,power_backbone,n_branches) 

        powers = get_branch_power_usage(power_model,input_shape,power_params)
        print('Branch powers:',powers)

        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.exp_name + "/weights_final.pth",map_location=device))
        model.set_inference_mode(args.width_mode)
        analysis_suffix = '/'+args.width_mode+'/'

    else:
        backbone = create_model(model_params)
        model = attach_branches(model_params,backbone,n_branches) 

        powers = get_branch_power_usage(model,input_shape,model_params)
        print('Branch powers:',powers)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.exp_name + "/weights_best.pth",map_location=device))
        analysis_suffix = ''

    print('Loaded Weights')
    model = model.to(device)
    model.eval()

    # create empty dict so that the other exit policies work
    decision_module_args = {}
    if args.exit_policy == 'decision_module':
        Decision_Modules,preprocessing_module = create_decision_module(model,device,pretrained = args.exp_name+'/decision_modules/'+args.decision_module_path+'/dm_best_weights.pth')
        decision_module_args['decision_modules'],decision_module_args['preprocessing_module'] = Decision_Modules,preprocessing_module

    if 'loss' not in model_params.keys():
        loss = 'cross-entropy'
    else:
        loss = model_params['loss']
        
    if loss == 'cross-entropy':
        output_function = nn.Softmax(dim=1)

    elif loss == 'dirichlet-loss':
        output_function = dirichlet_probability
    else:
        output_function = nn.Softmax(dim=1)

    labels = torch.zeros(n_values_test,dtype=int).to(device)
    logit_predictions = torch.zeros([n_values_test,n_branches,n_classes]).to(device)
    predictions = torch.zeros([n_values_test,n_branches,n_classes]).to(device)
    test_exits = torch.zeros([n_values_test,n_branches]).to(device)

    print('Obtaining Test Predictions')
    for test_idx, (x, y) in enumerate(tqdm(test_loader,desc='batch')):
        x,y = x.to(device),y.to(device)
        #1-forward pass - get logits
        with torch.no_grad():
            output,exits = get_output_with_exits(model,x,exit_policy=args.exit_policy,dm_args=decision_module_args)
            
        labels[test_idx*args.batch_size:test_idx*args.batch_size+len(y)] = y

        for branch in range(n_branches):
            logit_predictions[test_idx*args.batch_size:test_idx*args.batch_size+len(y),branch,:] = output[branch]
            predictions[test_idx*args.batch_size:test_idx*args.batch_size+len(y),branch,:] = output_function(output[branch])
            test_exits[test_idx*args.batch_size:test_idx*args.batch_size+len(y),branch] = exits[:,branch]
    
    os.makedirs(args.exp_name + "/analysis/"+analysis_suffix,exist_ok=True)
    save_directory = args.exp_name + "/analysis/"+analysis_suffix
    print('Saving Values')
   
    np.save(save_directory+'power_usage.npy',powers) 
    np.save(save_directory+'outputs.npy',predictions.cpu())    
    np.save(save_directory+'logit_outputs.npy',logit_predictions.cpu())    
    np.save(save_directory+'labels.npy',labels.cpu())

    os.makedirs(save_directory+'/exits/',exist_ok=True)
    if args.exit_policy == "decision_module":
        os.makedirs(save_directory+'/exits/decision_modules/'+args.decision_module_path,exist_ok=True)
        exit_directory = save_directory+'/exits/decision_modules/'+args.decision_module_path.strip('/')
        np.save(exit_directory+'.npy',test_exits.cpu())
    else:
        exit_directory = save_directory+'/exits/'
        np.save(exit_directory+args.exit_policy+'.npy',test_exits.cpu())

    os.makedirs(args.exp_name + "/analysis/callibration/"+analysis_suffix,exist_ok=True)
    save_directory = args.exp_name + "/analysis/callibration/"+analysis_suffix

    labels = torch.zeros(n_values_train,dtype=int).to(device)

    #raw output of model
    logit_predictions = torch.zeros([n_values_train,n_branches,n_classes]).to(device)
    #actual output of classifier
    predictions = torch.zeros([n_values_train,n_branches,n_classes]).to(device)
    train_exits = torch.zeros([n_values_train,n_branches]).to(device)

    print('Obtaining Train Predictions')
    for train_idx, (x, y) in enumerate(tqdm(train_loader,desc='batch')):
        x,y = x.to(device),y.to(device)
        #1-forward pass - get logits
        with torch.no_grad():
            output,exits = get_output_with_exits(model,x,exit_policy=args.exit_policy,dm_args=decision_module_args)

        labels[train_idx*args.batch_size:train_idx*args.batch_size+len(y)] = y

        for branch in range(n_branches):
            logit_predictions[train_idx*args.batch_size:train_idx*args.batch_size+len(y),branch,:] = output[branch]
            predictions[train_idx*args.batch_size:train_idx*args.batch_size+len(y),branch,:] = output_function(output[branch])
            train_exits[train_idx*args.batch_size:train_idx*args.batch_size+len(y),branch] = exits[:,branch]

    print('Saving Values')

    np.save(save_directory+'train_outputs.npy',predictions.cpu())    
    np.save(save_directory+'train_logit_outputs.npy',logit_predictions.cpu())    
    np.save(save_directory+'train_labels.npy',labels.cpu())

    #output exit probability -> entropy/decision_module
    
    os.makedirs(save_directory+'/exits/',exist_ok=True)
    if args.exit_policy == "decision_module":
        os.makedirs(save_directory+'/exits/decision_modules/'+args.decision_module_path,exist_ok=True)
        exit_directory = save_directory+'/exits/decision_modules/'+args.decision_module_path
        np.save(exit_directory+'/train.npy',train_exits.cpu())
    else:
        exit_directory = save_directory+'/exits/'
        np.save(exit_directory+args.exit_policy+'_train.npy',train_exits.cpu())

    labels = torch.zeros(n_values_val,dtype=int).to(device)
    logit_predictions = torch.zeros([n_values_val,n_branches,n_classes]).to(device)
    predictions = torch.zeros([n_values_val,n_branches,n_classes]).to(device)
    val_exits = torch.zeros([n_values_val,n_branches]).to(device)

    print('Obtaining Val Predictions')
    for val_idx, (x, y) in enumerate(tqdm(val_loader,desc='batch')):
        x,y = x.to(device),y.to(device)
        #1-forward pass - get logits
        with torch.no_grad():
            output,exits = get_output_with_exits(model,x,exit_policy=args.exit_policy,dm_args=decision_module_args)

        labels[val_idx*args.batch_size:val_idx*args.batch_size+len(y)] = y

        for branch in range(n_branches):
            logit_predictions[val_idx*args.batch_size:val_idx*args.batch_size+len(y),branch,:] = output[branch]
            predictions[val_idx*args.batch_size:val_idx*args.batch_size+len(y),branch,:] = output_function(output[branch])
            val_exits[val_idx*args.batch_size:val_idx*args.batch_size+len(y),branch] = exits[:,branch]

    print('Saving Values')

    np.save(save_directory+'val_outputs.npy',predictions.cpu())    
    np.save(save_directory+'val_logit_outputs.npy',logit_predictions.cpu())    
    np.save(save_directory+'val_labels.npy',labels.cpu())

    #output exit probability -> entropy/decision_module
    os.makedirs(save_directory+'/exits/',exist_ok=True)
    if args.exit_policy == "decision_module":
        os.makedirs(save_directory+'/exits/decision_modules/'+args.decision_module_path,exist_ok=True)
        exit_directory = save_directory+'/exits/decision_modules/'+args.decision_module_path
        np.save(exit_directory+'/val.npy',val_exits.cpu())
    else:
        exit_directory = save_directory+'/exits/'
        np.save(exit_directory+args.exit_policy+'_val.npy',val_exits.cpu())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Directory to obtain predictions from",type=str)
    parser.add_argument("-ds","--data", help="Which dataset to use",type=str,default='CIFAR10')
    parser.add_argument("-ba","--batch_size", help="Batch size for training",type=int,default=128)    
    parser.add_argument("-wm","--width_mode", help="If using slimmable net define width mode ('full', 'half', 'quarter'). Defaults to 'full'",type=str,default='full')    
    parser.add_argument("-ex","--exit_policy", help="Exit policy to obtains exits for",type=str,default='entropic')    
    parser.add_argument("-dd","--decision_module_path", help="If using a decision module for the exit policy, the directory to take the weights from",type=str,default='none')    


    args = parser.parse_args()
    main(args)