import os

import torch
from torch import nn,optim

import numpy as np
import argparse
from tqdm import tqdm
import json

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.get_model import create_model,attach_branches
from utils.get_data import get_dataset,preprocessing_empty,preprocessing_regmixup

from utils.utils import branch_ce_loss, branch_dirichlet_loss, branch_dirichlet_ensemble_loss, branch_metric_loss, branch_regmixup_loss, get_branched_accuracies
from utils.analysis_utils import set_unique_exp_name

'''
TODO: incorporate train_loop into this script
'''

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
        
    backbone = create_model(input_params)
    model = attach_branches(input_params,backbone,n_branches = args.n_branches)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # ce_loss = nn.CrossEntropyLoss()

    #Defining optimiser and scheduler
    params = model.parameters()
    optimiser = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimiser = optim.Adam(params, lr=args.learning_rate, weight_decay=5e-5, betas=[0.9,0.99])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,T_max=100)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,patience=50,factor=0.1)

    #for tracking best model
    best_loss = np.log(classes)

    #defining our branched loss
    if args.loss == 'cross-entropy':
        weight_dict = {'final_weight':0.4}
        branched_loss = branch_ce_loss(weights='weighted-final',n_branches=args.n_branches,**weight_dict)
        directory_identfier = ''
        loss_preprocessing = preprocessing_empty
    elif args.loss == 'dirichlet-loss':
        weight_dict = {'final_weight':0.4}
        annealing_step = (data_props['n_values_train']/args.batch_size)*300 #N_steps per epoch mutliplied by 100
        branched_loss = branch_dirichlet_loss(weights='weighted-final',n_branches=args.n_branches,n_classes = classes, global_step=0, annealing_step=annealing_step, **weight_dict)
        directory_identfier = '-dirichlet-loss'
        loss_preprocessing = preprocessing_empty

    elif args.loss == 'dirichlet-ensemble-loss':
        weight_dict = {'final_weight':0.4}
        annealing_step = (data_props['n_values_train']/args.batch_size)*300 #N_steps per epoch mutliplied by 100
        branched_loss = branch_dirichlet_ensemble_loss(weights='weighted-final',n_branches=args.n_branches,n_classes = classes, global_step=0, annealing_step=annealing_step, **weight_dict)
        directory_identfier = '-dirichlet-ensemble-loss'
        loss_preprocessing = preprocessing_empty

    elif args.loss == 'metric-loss':
        weight_dict = {'final_weight':0.4}
        annealing_step = (data_props['n_values_train']/args.batch_size)*10 #N_steps per epoch mutliplied by when we want the mistake metric to take effect
        branched_loss = branch_metric_loss(weights='weighted-final',n_branches=args.n_branches,n_classes = classes, global_step=0, annealing_step=annealing_step, **weight_dict)
        directory_identfier = '-metric-loss'
        loss_preprocessing = preprocessing_empty

    elif args.loss == 'regmixup':
        weight_dict = {'final_weight':0.4}
        branched_loss = branch_regmixup_loss(weights='weighted-final',n_branches=args.n_branches,**weight_dict)
        directory_identfier = '-regmixup-loss'
        loss_preprocessing = preprocessing_regmixup

    branched_loss.to(device)

    os.makedirs('./trained-models/'+args.data+'/'+args.backbone+'_'+ str(args.n_branches)+'_branch/w'+str(args.width)+'_d'+str(args.depth)+'/',exist_ok=True)
    save_directory = './trained-models/'+args.data+'/'+args.backbone+'_'+ str(args.n_branches)+'_branch/w'+str(args.width)+'_d'+str(args.depth)+'/'
    directory = set_unique_exp_name(save_directory,args.exp_name+directory_identfier)
    
    # Serialize data into file:
    with open(directory+"/model_params.json", 'w' ) as f: 
        json.dump(input_params, f)
        
    writer = SummaryWriter(log_dir=directory)

    print('beginning training')
    
    for epoch in tqdm(range(args.epochs),desc='Epochs'):
        train_losses = np.zeros(len(train_loader))
        train_branch_losses = np.zeros((len(train_loader),args.n_branches))
        train_accuracies = np.zeros((len(train_loader),args.n_branches))
        model.train()
        for train_idx, (x, y) in enumerate(train_loader):
            train_step = epoch*len(train_loader) + train_idx
            x,y,original_y = loss_preprocessing(x.to(device),y.to(device))

            #1-forward pass - get logits
            output = model(x)

            #2-objective function
            J_train,batch_branch_losses = branched_loss(output,y)
            batch_train_accuracies = get_branched_accuracies(output,original_y)
            
            #3-clean gradients
            model.zero_grad()
            
            #4-accumulate partial derivatives of J
            J_train.backward()
            
            #5-write metrics to summary file
            train_losses[train_idx] = J_train
            writer.add_scalar("Step total loss train",J_train,train_step)
            for idx,branch_loss in enumerate(batch_branch_losses):
                train_branch_losses[train_idx,idx] = branch_loss
                train_accuracies[train_idx,idx] = batch_train_accuracies[idx]
                writer.add_scalar(("Branch Losses/Branch "+str(idx)),branch_loss,train_step)
                writer.add_scalar(("Branch Accuracies/Branch  "+str(idx)),batch_train_accuracies[idx],train_step)

            #6-step in opposite direction of gradient
            optimiser.step()

        model.eval()
        val_losses = np.zeros(len(val_loader))
        val_branch_losses = np.zeros((len(val_loader),args.n_branches))
        val_accuracies = np.zeros((len(val_loader),args.n_branches))
        for val_idx, (x, y) in enumerate(val_loader):
            val_step = epoch*len(val_loader) + val_idx
            x,y,original_y = loss_preprocessing(x.to(device),y.to(device))
            #1-forward pass - get logites
            with torch.no_grad():
                output = model(x)

            #2-metrics
            J_val,batch_branch_losses = branched_loss(output,y)
            validation_accuracies_batch = get_branched_accuracies(output,original_y)
            
            # J_val = 0.25*ce_loss(output[0],y)  + 0.25*ce_loss(output[1],y)  + 0.25*ce_loss(output[2],y)  + 0.25*ce_loss(output[3],y)
            
            #5-write metrics to summary file
            val_losses[val_idx] = J_val
            
            writer.add_scalar("Step total loss train",J_val,val_step)
            for idx,branch in enumerate(batch_branch_losses):
                val_branch_losses[val_idx,idx] = branch_loss
                val_accuracies[val_idx,idx] = validation_accuracies_batch[idx]
                writer.add_scalar(("Branch validation losses/Branch " +str(idx)),branch,val_step)
                writer.add_scalar(("Branch validation Accuracies/Branch "+str(idx)),validation_accuracies_batch[idx],val_step)

        epoch_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if epoch_loss < best_loss:
            torch.save(model.state_dict(), directory + '/weights_best.pth')
            best_loss = epoch_loss

        writer.add_scalar("Epoch loss train",epoch_loss,epoch)
        writer.add_scalar("Epoch loss val",val_loss,epoch)

        mean_train_branch_losses = train_branch_losses.mean(axis=0)
        mean_val_branch_losses = val_branch_losses.mean(axis=0)

        mean_train_accuracies = train_accuracies.mean(axis=0)
        mean_val_accuracies = val_accuracies.mean(axis=0)

        for idx in range(args.n_branches):
            writer.add_scalar(("Branch epoch losses/Branch "+str(idx)),mean_train_branch_losses[idx],epoch)
            writer.add_scalar(("Branch epoch accuracies/Branch "+str(idx)),mean_train_accuracies[idx],epoch)

            writer.add_scalar(("Branch epoch validation losses/Branch "+str(idx)),mean_val_branch_losses[idx],epoch)
            writer.add_scalar(("Branch validation epoch accuracies/Branch "+str(idx)),mean_val_accuracies[idx],epoch)
        
        scheduler.step(val_loss)

        if epoch%10 == 0:
            print('epoch: ',epoch,', val_accs: ',mean_val_accuracies, ', train_accs: ',mean_train_accuracies)

    writer.close()
    event_acc = EventAccumulator(directory)
    event_acc.Reload()

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    # _, step_nums, vals = zip(*event_acc.Scalars('Epoch loss val'))
    # print(step_nums,vals)

    torch.save(model.state_dict(), directory + '/weights_epoch_'+str(epoch)+'.pth')

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
    parser.add_argument("-l","--loss", help="loss function for training",type=str,default='cross-entropy')

    parser.add_argument("-rd", "--resolution_dependent", help="first layer resolution dependence",type=bool,default=False)

    args = parser.parse_args()
    main(args)
        

