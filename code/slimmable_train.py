import os

import torch
from torch import nn,optim

import numpy as np
import argparse
from tqdm import tqdm
import json

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.utils import train_loop
from utils.get_model import create_model,attach_branches
from utils.get_data import get_dataset

from utils.utils import branch_ce_loss
from utils.analysis_utils import set_unique_exp_name


def main(args):
    input_params = vars(args)

    train_loader,val_loader,_,data_props = get_dataset(args.data,args.batch_size)
    n_channels = data_props['n_channels']
    classes = data_props['classes']

    input_params['n_classes'] = classes
    input_params['in_channels'] = n_channels
    input_params['n_branches'] = args.n_branches

    if args.resolution_dependent:
        input_params['res'] = data_props['res']
    else:
        input_params['res'] = 224 

    input_params['backbone'] = 'slimmable_'+input_params['backbone']

    backbone = create_model(input_params)
    model = attach_branches(input_params,backbone,n_branches = args.n_branches) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Defining optimiser and scheduler
    params_quarter = model.parameters()
    params_half = model.parameters()
    params_full = model.parameters()
    # optimiser = optim.Adam(params, lr=args.learning_rate, weight_decay=5e-5, betas=[0.9,0.99])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,T_max=100)

    #optimiser and scheduler for each split of the model
    optimiser_quarter = optim.SGD(params_quarter, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimiser_half = optim.SGD(params_half, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimiser_full = optim.SGD(params_full, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler_quarter = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_quarter,patience=10,factor=0.1)
    scheduler_half = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_half,patience=10,factor=0.1)
    scheduler_full = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_full,patience=10,factor=0.1)

    os.makedirs('./trained-models/'+args.data+'/'+args.backbone+'_'+ str(args.n_branches)+'_branch/w'+str(args.width)+'_d'+str(args.depth)+'/',exist_ok=True)
    save_directory = './trained-models/'+args.data+'/'+args.backbone+'_'+ str(args.n_branches)+'_branch/w'+str(args.width)+'_d'+str(args.depth)+'/'
    directory = set_unique_exp_name(save_directory,args.exp_name)
    
    # Serialize data into file:
    with open(directory+"/model_params.json", 'w' ) as f: 
        json.dump(input_params, f)
        
    writer = SummaryWriter(log_dir=directory)

    #for tracking best model
    best_loss = np.log(classes)

    #defining our branched loss
    weight_dict = {'final_weight':0.4}
    branched_loss = branch_ce_loss(weights='weighted-final',n_branches=args.n_branches,**weight_dict)
    branched_loss.to(device)

    train_utils = {}
    train_utils['dataloaders'] = {'train':train_loader,'val':val_loader}
    train_utils['optimisation'] = {'loss':branched_loss,'best_loss':best_loss}
    train_utils['logging'] = {'writer':writer,'directory':directory}
    
    print('beginning training')
    train_utils['optimisation']['total_epochs'] = 0
    train_schedule = list()

    train_schedule.append({'switch':'full','epochs':150})
    train_schedule.append({'switch':'middle','epochs':100})
    train_schedule.append({'switch':'quarter','epochs':150})

    train_schedule.append({'switch':'half-fine','epochs':50})
    train_schedule.append({'switch':'full-fine','epochs':100})

    #train loop to accept training schedule
    for phase,hparams in enumerate(train_schedule):
        #assign optimisers
        switch = hparams['switch']
        epochs = hparams['epochs']
        print('phase: ',phase+1,'\tswitch:',switch,'\tepochs:',epochs)
        if switch == 'quarter' or switch == 'quarter-fine':
            #train quarter
            train_utils['optimisation']['scheduler'], train_utils['optimisation']['optimiser'] = scheduler_quarter,optimiser_quarter
        elif switch == 'half' or switch == 'half-fine':
            #train half
            train_utils['optimisation']['scheduler'], train_utils['optimisation']['optimiser'] = scheduler_half,optimiser_half
        elif switch == 'full' or switch == 'full-fine':
            #train full
            train_utils['optimisation']['scheduler'], train_utils['optimisation']['optimiser'] = scheduler_full,optimiser_full

        #setup model
        model.set_training_state(width=switch)
        train_utils['logging']['writer_prefix'] = '/'+switch+'/'

        #train model
        args.epochs = epochs

        model,train_utils = train_loop(args,model,device,train_utils)

        #save model
        total_epochs = train_utils['optimisation']['total_epochs']
        torch.save(model.state_dict(), directory + '/weights_epoch_'+str(total_epochs)+'.pth')


    torch.save(model.state_dict(), directory + '/weights_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Name of run",type=str,default='unnamed')
    parser.add_argument("-bck","--backbone", help="Backbone architecture to be used",type=str,default='resnet')
    parser.add_argument("-dpth","--depth", help="Backbone architecture to be used",type=int,default=18)
    parser.add_argument("-wdth","--width", help="Backbone architecture to be used",type=float,default=1.)
    parser.add_argument("-nb","--n_branches", help="Backbone architecture to be used",type=int,default=4)

    parser.add_argument("-ds","--data", help="Which dataset to use",type=str,default='CIFAR10')
    parser.add_argument("-ep","--epochs", help="Number of epochs to run experiment for",type=int,default=100)
    parser.add_argument("-ba","--batch_size", help="Batch size for training",type=int,default=128)
    parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=float,default=1e-2)

    parser.add_argument("-rd", "--resolution_dependent", help="first layer resolution dependence",type=bool,default=True)

    args = parser.parse_args()
    main(args)
        

