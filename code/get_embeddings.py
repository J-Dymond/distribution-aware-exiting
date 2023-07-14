import os
import random
import sys
import tracemalloc

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import json

from utils.get_model import load_pretrained
from utils.get_data import get_dataset
from utils.analysis_utils import stack_embeddings
from utils.knn_utils import get_penultimate_input


final_activation_hook = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        final_activation_hook[name] = output.detach()
    return hook

def main(args):
    model_param_file = open(args.exp_name + '/model_params.json')
    model_params = json.load(model_param_file)
    n_branches = model_params['n_branches']

    train_loader,val_loader,test_loader,data_props = get_dataset(args.data,batch_size=args.batch_size,shuffling=False)

    if args.partition.lower() == 'train':
        data_loader = train_loader
    elif args.partition.lower() == 'val':
        data_loader = val_loader
    elif args.partition.lower() == 'test':
        data_loader = test_loader
    else:
        print('invalid partition specified, using validation set')
        data_loader = val_loader

    n_values_train = data_props['n_values_train']
    n_values_val = data_props['n_values_val']

    n_classes = data_props['classes']
    res = data_props['res']

    if 'res' not in model_params.keys():
        model_params['res'] = res

    n_channels = data_props['n_channels']
    input_shape = ((n_channels,res,res))

    save_directory = args.exp_name+'/embeddings/'+args.data+'/'+args.partition+'/'
    print('saving batches to:',save_directory)
    os.makedirs(save_directory,exist_ok=True)
    for branch_idx in range(n_branches):
        os.makedirs((save_directory+'/branch_'+str(branch_idx)+'/'),exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_model = load_pretrained(args.exp_name,model_params,data_props,device,weights_file=args.weights_file).to(device)
    backbone_model.eval()
    hook1 = backbone_model.backbone.encoder.blocks[-1].blocks[-1].register_forward_hook(getActivation('encoder_output'))

    #put modules on device
    preprocessing_module = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(start_dim=1)).to(device)

    get_input = get_penultimate_input(backbone_model,preprocessing_module,final_activation_hook)

    #we normalise in knn script
    normalise = nn.functional.normalize

    all_out = list()
    for branch_idx, _ in enumerate(range(n_branches)):
        all_out.append(list())

    save_idx = 0
    for batch_idx, (x,y) in enumerate(tqdm(data_loader)):
        for branch_idx, contrastive_module in enumerate(range(n_branches)):
            x,y = x.to(device),y.to(device)
            batch_output = get_input(x,branch_idx)

            all_out[branch_idx].append(batch_output)
    
        if batch_idx % 5 == 0:
            print('saving at batch:',batch_idx)
            for branch_idx, _ in enumerate(range(n_branches)):
                branch_array = torch.vstack(all_out[branch_idx])
                print('branch:',branch_idx,'\tshape:',branch_array.shape)            
                torch.save(branch_array,(save_directory+'/branch_'+str(branch_idx)+'/batch_'+str(save_idx)+'.pt'))
            all_out = list()
            for branch_idx, _ in enumerate(range(n_branches)):
                all_out.append(list())
            save_idx=save_idx+1

    #save last batch of embeddings
    print('saving at batch:',batch_idx)
    for branch_idx, _ in enumerate(range(n_branches)):
        branch_array = torch.vstack(all_out[branch_idx])
        print('branch:',branch_idx,'\tshape:',branch_array.shape)            
        torch.save(branch_array,(save_directory+'/branch_'+str(branch_idx)+'/batch_'+str(save_idx)+'.pt'))
            

    hook1.remove()

    if args.stacking:
        stack_embeddings(args.exp_name,args.data,args.partition)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Directory to obtain predictions from",type=str)
    parser.add_argument("-wf","--weights_file", help="Weights file to load weights from",type=None)
    parser.add_argument("-d","--data", help="Which dataset to use",type=str,default='CIFAR10')
    parser.add_argument("-p","--partition", help="Which dataset to use",type=str,default='train')
    parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)    
    parser.add_argument("-w","--width_mode", help="If using slimmable net define width mode ('full', 'half', 'quarter'). Defaults to 'full'",type=str,default='full')    
    parser.add_argument("-st","--stacking", help="Stacking embdeddings afterwards. True or False",type=bool,default=True)    


    args = parser.parse_args()
    main(args)