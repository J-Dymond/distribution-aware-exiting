import os
import re

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import json

from utils.get_model import load_pretrained
from utils.get_data import get_dataset
from utils.analysis_utils import load_embeddings
from utils.confidence_utils import get_nearest_k_func
from utils.knn_utils import get_penultimate_input

final_activation_hook = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        final_activation_hook[name] = output
    return hook

def main(args):
    model_param_file = open(args.exp_name + '/model_params.json')
    model_params = json.load(model_param_file)
    n_branches = model_params['n_branches']

    train_loader,val_loader,test_loader,data_props = get_dataset(args.data_test,batch_size=args.batch_size,shuffling=False)

    if args.partition_test.lower() == 'train':
        data_loader = train_loader
    elif args.partition_test.lower() == 'val':
        data_loader = val_loader
    elif args.partition_test.lower() == 'test':
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

    save_directory = args.exp_name+'/knn_ood/train_'+args.data_train+'_'+args.partition_train+'/test_'+args.data_test+'_'+args.partition_test+'/'
    print('saving batches to:',save_directory)
    os.makedirs(save_directory,exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_model = load_pretrained(args.exp_name,model_params,data_props,device,weights_file=args.weights_file).to(device)
    backbone_model.eval()
    # hook1 = backbone_model.backbone.decoder.avg.register_forward_hook(getActivation('avgpool'))
    hook1 = backbone_model.backbone.encoder.blocks[-1].blocks[-1].register_forward_hook(getActivation('encoder_output'))

    #put modules on device
    preprocessing_module = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(start_dim=1)).to(device)
    get_embedding = get_penultimate_input(backbone_model,preprocessing_module,final_activation_hook)
    normalise = nn.functional.normalize
    # starting the monitoring

    train_embeddings = load_embeddings(args.exp_name,args.data_train,args.partition_train)

    for branch_idx,embedding in enumerate(train_embeddings):
        print('Branch:',str(branch_idx),'\tembedding shape:', embedding.shape)
        train_embeddings[branch_idx] = embedding/torch.norm(embedding,dim=0)

    # k_vals = torch.zeros((n_values_val,n_branches,args.k))

    for batch_idx, (x,y) in enumerate(tqdm(data_loader)):
        batch_k_vals = torch.zeros((len(y),n_branches,args.k))
        for branch_idx, _ in enumerate(range(n_branches)):
            branch_kvals = torch.zeros((len(y),args.k))
            x,y = x.to(device),y.to(device)
            with torch.no_grad():
                batch_output = get_embedding(x,branch_idx)
            
            for mini_batch_idx,test_embedding in enumerate(batch_output):
                nearest_k = get_nearest_k_func(train_embeddings[branch_idx],test_embedding,k=args.k)
                branch_kvals[mini_batch_idx,:] = nearest_k
            batch_k_vals[:,branch_idx,:] = branch_kvals
        
        torch.save(batch_k_vals,save_directory+'/batch_'+str(batch_idx)+'.pt')


    all_vals_list = list()
    for file in os.listdir(save_directory):
        if re.search(r'batch_\d+.*\.pt$', file):
            batch = torch.load(save_directory+'/'+file)
            all_vals_list.append(batch)

    all_vals_tensor = torch.vstack(all_vals_list).detach().numpy()
    print('saving k nearest neighbour array of size:',all_vals_tensor.shape)

    np.save((save_directory+'all_k_distances.npy'),all_vals_tensor)
    np.save((save_directory+'random_seed.npy'),torch.seed())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Directory to obtain predictions from",type=str)
    parser.add_argument("-wf","--weights_file", help="Weights file to load weights from",type=None)

    parser.add_argument("-k","--k",help="How many samples to use in KNN algorithm",type=int,default=50)

    parser.add_argument("-dtr","--data_train", help="Which dataset to use",type=str,default='CIFAR10')
    parser.add_argument("-dte","--data_test", help="Which dataset to use",type=str,default='CIFAR100')

    parser.add_argument("-ptr","--partition_train", help="Which dataset to use",type=str,default='train')
    parser.add_argument("-pte","--partition_test", help="Which dataset to use",type=str,default='test')

    parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)    
    parser.add_argument("-w","--width_mode", help="If using slimmable net define width mode ('full', 'half', 'quarter'). Defaults to 'full'",type=str,default='full')    

    args = parser.parse_args()
    main(args)