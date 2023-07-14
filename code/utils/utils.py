from sched import scheduler
import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical 

#This allows code to correctly calculate the convolutional operations of the custom conv module
from models.resnet import Conv2dAuto
from ptflops.flops_counter import get_model_complexity_info, conv_flops_counter_hook

import numpy as np 
import re
from tqdm import tqdm

slimmable_dict = {'full':1.0,'half':0.5,'quarter':0.25}

## For implementing dirichlet loss
def KL(alpha,beta):
    torch.transpose(alpha,0,1)
    S_alpha = torch.sum(alpha,dim=1,keepdim=True)
    S_beta = torch.sum(beta,dim=1,keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1,keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta),dim=1,keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta)*(dg1-dg0),dim=1,keepdim=True) + lnB + lnB_uni
    return kl

def dirichlet_mse_loss(target, alpha, beta, global_step, annealing_step, n_classes): 
    p = F.one_hot(target, num_classes=n_classes)
    S = torch.sum(alpha, 1, keepdim=True) 
    E = alpha - 1
    m = alpha / S


    A = torch.sum((p-m)**2, 1,keepdim=True) 
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)),1,keepdim=True) 
    # B = torch.sum(p*(1-p)/(S+1),1,keepdim=True) 

    annealing_coef = torch.min(torch.tensor([1.0,global_step/annealing_step]))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp,beta)

    return torch.mean((A + B) + C)

class custom_relu(nn.Module):
    def __init__(self,positive_coef=1,negative_coef=0):
        super().__init__()
        self.pos_c = positive_coef
        self.neg_c = negative_coef
    def forward(self,input):
        positive = (input>0).float()
        negative = positive.clone() - 1
        coefficients = self.pos_c*positive + self.neg_c*negative
        output = input*coefficients
        return output

#deprecated functions
def dirichlet_probability(logits):
    alpha = F.relu(logits) + 1
    S = torch.sum(alpha, 1, keepdim=True) 
    probs = alpha/S
    return probs

def dirichlet_ensemble_probability(branch_outputs,branch_idx):
    ensemble_logits = torch.stack(branch_outputs[:(branch_idx+1)])
    alpha = F.relu(ensemble_logits) + 1
    S = torch.sum(alpha, 1, keepdim=True) 
    probs = alpha/S
    return probs

"""
TODO: implement below function into the two branched loss functions
"""

def get_branch_weights(weights,n_branches,**kwargs):
    if weights == 'even':
            placeholder = torch.full([n_branches],(1/n_branches))

    elif weights == 'weighted-final':
        assert weights, 'to use final weighted layer, provide a weight for final layer: final_weight'
        final_weight = kwargs['final_weight']
        other_values = (1-final_weight)/(n_branches-1)
        placeholder = torch.full([n_branches],other_values)
        placeholder[-1] = final_weight
    
    elif weights == 'custom':
        assert kwargs['weightings'], 'to use custom weightings, provide a weights array of length n_branches: weightings'
        assert len(kwargs['weightings']) == n_branches, 'to use custom weightings, provide a weights array of length n_branches: weightings'
        placeholder = torch.tensor(kwargs['weightings'])
    
    else:
        placeholder = torch.full([n_branches],(1/n_branches))
    
    return(placeholder)

class branch_metric_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, n_classes = 10, global_step = 1, annealing_step = 1000, **kwargs):
        super().__init__()

        self.n_classes = n_classes
        #we use this to gradually increase low confidence error requirement
        self.global_step = global_step
        self.annealing_step = annealing_step
        
        self.max_entropy = torch.log(torch.tensor(n_classes))
        self.ce_loss = nn.CrossEntropyLoss()
        self.output_prob = nn.Softmax(dim=1)

        self.loss_normaliser = custom_relu(positive_coef=1,negative_coef=1)

        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, outputs, target):

        losses = list()
        loss = self.init_loss
        annealing_coef = torch.min(torch.tensor([1.0,self.global_step/self.annealing_step]))

        for idx,branch in enumerate(outputs):
            entropies = Categorical(logits=torch.abs(branch-1e-15)).entropy()/self.max_entropy

            accuracies = torch.eq(torch.argmax(branch,dim=1),target).int()

            #penalise confident mistakes
            mistakes = (accuracies - 1)*(-1)
            incorrect_metric = -1*mistakes*(entropies)
            
            #make the minimum loss -1 the max 1 and center it around 0
            ce_los = (-1+self.ce_loss(branch.float(),target)/(self.max_entropy/2))
            #normaliser accentuates positive and negative components differently
            correct_metric = self.loss_normaliser(ce_los)

            #combine losses
            branch_losses = correct_metric + annealing_coef*incorrect_metric
            branch_loss = torch.mean(branch_losses)
            
            #create branch loss
            losses.append(branch_loss) 
            loss = loss + self.weighting[idx]*branch_loss

        #iterate for annealling coefficient
        iteration_placeholder = self.global_step+1
        self.global_step = iteration_placeholder

        return(loss,losses)

    def get_probs(self, output):
        probs = list()
        for branch_logits in output:
            probs.append(self.output_prob(branch_logits))
        return probs

class branch_dirichlet_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, n_classes = 10, global_step = 0, annealing_step = 1000, **kwargs):
        super().__init__()

        self.logits_to_evidence = nn.ReLU()
        self.n_classes = n_classes
        self.dirichlet_loss = dirichlet_mse_loss
        self.annealing_step = annealing_step
        self.global_step = global_step

        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("beta", torch.ones((1,n_classes)))
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, outputs, target):

        losses = list()
        loss = self.init_loss

        for idx,branch in enumerate(outputs):
            evidence = self.logits_to_evidence(branch)
            alpha = evidence + 1
            branch_loss = self.dirichlet_loss(target,alpha,self.beta,self.global_step,self.annealing_step,self.n_classes)
            losses.append(branch_loss)
            loss = loss + self.weighting[idx]*branch_loss

        iteration_placeholder = self.global_step+1
        self.global_step = iteration_placeholder

        return(loss,losses)
    
    def get_probs(self, output):
        probs = list()
        for branch_logits in output:
            alpha = F.relu(branch_logits) + 1
            S = torch.sum(alpha, 1, keepdim=True) 
            probs.append(alpha/S)
        return probs

class branch_dirichlet_ensemble_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, n_classes = 10, global_step = 0, annealing_step = 1000, **kwargs):
        super().__init__()

        self.logits_to_evidence = nn.ReLU()
        self.n_classes = n_classes
        self.dirichlet_loss = dirichlet_mse_loss
        self.annealing_step = annealing_step
        self.global_step = global_step

        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("beta", torch.ones((1,n_classes)))
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, outputs, target):

        losses = list()
        evidence_stack = list()
        loss = self.init_loss

        for idx,branch in enumerate(outputs):
            #get branch evidence
            evidence = self.logits_to_evidence(branch)
            #accumulate total evidence
            evidence_stack.append(evidence)
            ensemble_evidence = torch.sum(torch.stack(evidence_stack),dim=0)
            #get alpha
            ensemble_alpha = ensemble_evidence + 1
            #pass to loss
            branch_loss = self.dirichlet_loss(target,ensemble_alpha,self.beta,self.global_step,self.annealing_step,self.n_classes)
            losses.append(branch_loss)
            loss = loss + self.weighting[idx]*branch_loss

        iteration_placeholder = self.global_step+1
        self.global_step = iteration_placeholder

        return(loss,losses)

    def get_probs(self,output):
        probs = list()
        evidence_stack = list()
        for branch_output in output:
            evidence = self.logits_to_evidence(branch_output)
            #accumulate total evidence
            evidence_stack.append(evidence)
            ensemble_evidence = torch.sum(torch.stack(evidence_stack),dim=0)
            alpha = F.relu(ensemble_evidence) + 1
            S = torch.sum(alpha, 1, keepdim=True) 
            probs.append(alpha/S)
        return probs

class branch_ce_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, **kwargs):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.output_prob = nn.Softmax(dim=1)
        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, outputs, target):
        losses = list()
        loss = self.init_loss
        for idx,branch in enumerate(outputs):
            branch_loss = self.ce_loss(branch,target)
            losses.append(branch_loss)
            loss = loss + self.weighting[idx]*branch_loss
        return(loss,losses)
    
    def get_probs(self, output):
        probs = list()
        for branch_logits in output:
            probs.append(self.output_prob(branch_logits))
        return probs

def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class branch_regmixup_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, **kwargs):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.output_prob = nn.Softmax(dim=1)
        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, logits, targets):
        (targets_a,targets_b,lam) = targets
        losses = list()
        loss = self.init_loss
        for idx,branch in enumerate(logits):
            branch_loss = regmixup_criterion(self.ce_loss,branch,targets_a,targets_b,lam)
            losses.append(branch_loss)
            loss = loss + self.weighting[idx]*branch_loss
        return(loss,losses)
    
    def get_probs(self, output):
        probs = list()
        for branch_logits in output:
            probs.append(self.output_prob(branch_logits))
        return probs


class branch_supcon_loss(nn.Module):
    def __init__(self, weights='even', n_branches = 4, **kwargs):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.output_prob = nn.Softmax(dim=1)
        placeholder = get_branch_weights(weights,n_branches,**kwargs)
        
        self.register_buffer("weighting", placeholder)
        self.register_buffer("init_loss", torch.tensor([0.0], requires_grad=True))

    def forward(self, logits, targets):
        (targets_a,targets_b,lam) = targets
        losses = list()
        loss = self.init_loss
        for idx,branch in enumerate(logits):
            branch_loss = regmixup_criterion(self.ce_loss,branch,targets_a,targets_b,lam)
            losses.append(branch_loss)
            loss = loss + self.weighting[idx]*branch_loss
        return(loss,losses)
    
    def get_probs(self, output):
        probs = list()
        for branch_logits in output:
            probs.append(self.output_prob(branch_logits))
        return probs



def get_branched_accuracies(outputs,target):
    accuracies = list()
    for branch in outputs:
        branch = branch[:len(target)]
        accuracies.append(target.eq(branch.detach().argmax(dim=1)).float().mean())
    return(accuracies)

def get_branch_power_usage(model,input_shape,model_params):
    model_name = model_params['backbone']
    if model_name == 'resnet':
        get_layers = get_layers_resnet
    
    _,_,power_usage = get_model_complexity_info(model,input_res=input_shape,custom_modules_hooks={Conv2dAuto:conv_flops_counter_hook})

    macs = np.array(power_usage[0])
    params = np.array(power_usage[1])
    names = np.array(power_usage[2])

    full_model = macs[0]

    layers = get_layers(names)

    names = names[layers]
    macs = macs[layers]
    params = params[layers]

    branch_points = model.selected_layers
    n_branches = model_params['n_branches']

    powers = list()
    for idx, branch in enumerate(branch_points):
        branch_name = 'backbone.'+branch.strip('.conv')
        value = np.argwhere(names == branch_name)[0][0]
        power = sum(macs[n_branches-1:value+1]) + sum(macs[1:idx+1])
        powers.append(power)

    powers.append(full_model)

    return(powers)

def get_layers_resnet(names):
    layers = list()
    for idx,name in enumerate(names):
        block_count = len(re.findall('blocks',name))
        gate = re.search('gate.[0-9]$',name)
        if block_count > 2 and re.search('.[0-9]$',name):
                layers.append(idx)
        elif re.search('branches.[0-9]$',name):
            layers.append(idx)
        elif block_count == 2 and re.search('.shortcut$',name):
            layers.append(idx)
        elif name == 'backbone.decoder':
            layers.append(idx)
        elif gate:
            layers.append(idx)

    return(layers)

    
def get_branch_input(model,preprocessing_module,branch):
    #build input, not very efficient at the moment
    branches = list(model.selected_out.items())[:branch+1]
    branch_input_list = list()
    for inp in branches:
        branch_input_list.append(preprocessing_module(inp[1]))
    branch_input = torch.hstack(branch_input_list)
    return branch_input

def get_output_with_exits(model,inputs,exit_policy='entropic',dm_args={}):
    output = model(inputs)
    exits = torch.zeros((len(inputs),len(output)))

    if exit_policy == 'entropic':
        max_entropy = output[0].shape[1]
        for branch_idx,branch_out in enumerate(output):
            exits[:,branch_idx] = Categorical(logits=branch_out).entropy()/max_entropy

    elif exit_policy == 'max_prob':
        max_prob = len(output)
        max_entropy = output[0].shape[1]
        for branch_idx,branch_out in enumerate(output):
            exits[:,branch_idx] = torch.max(nn.functional.softmax(branch_out,dim=1),dim=1)[0]/max_prob

    elif exit_policy == 'cumulative_prob':
        outputs = list()
        for branch_idx,branch_out in enumerate(output):
            outputs.append(branch_out)
            policy_input = nn.functional.softmax(torch.sum(torch.dstack(outputs),dim=2),dim=1)
            exits[:,branch_idx] = torch.max(policy_input,dim=1)[0]

    elif exit_policy == 'mutual_agreement':
        previous_preds = torch.full(inputs.shape[0],-1)
        for branch_idx,branch_out in enumerate(output):
            entropies = Categorical(logits=branch_out).entropy()/max_entropy
            predictions = torch.argmax(policy_input,dim=1)[0]
            previous_preds[:] = torch.argmax(policy_input,dim=1)[0]
            matching = torch.tensor(predictions==previous_preds,dtype=int)*exits
            exits[:,branch_idx] = torch.max(entropies,matching,dim=0)
            
            
    elif exit_policy == 'decision_module':
        decision_modules = dm_args['decision_modules']
        preprocessing_module = dm_args['preprocessing_module']
        #build input, not very efficient at the moment
        for branch_idx,Decision_Module in enumerate(decision_modules):
            branch_input = get_branch_input(model,preprocessing_module,branch_idx)
            #get output prediction of decision module
            exits[:,branch_idx] = Decision_Module(branch_input)
    
    return(output,exits)

def train_loop(input_args,model,device,train_utils,return_best=True):
    '''
    This is the basic training loop for training the models

    Parameters
    -------------------
    input_args: this is the dictionary of input arguments for the training phase
    model: the model to be used for training
    train_utils: the utilities needed for training such as loss function, optimiser, dataloaders etc.. 
    return_best: flag which defaults to True, determines whether the best model found in the training phase is returned, or the most recent one 

    Outputs
    -------------------
    model: The model with trained weights found during training phase
    train_utils: Updated training utilities 
    '''
    
    #loading utils
    train_loader,val_loader = train_utils['dataloaders']['train'], train_utils['dataloaders']['val']
    branched_loss = train_utils['optimisation']['loss']
    best_loss, total_epochs = train_utils['optimisation']['best_loss'], train_utils['optimisation']['total_epochs']
    optimiser,scheduler = train_utils['optimisation']['optimiser'], train_utils['optimisation']['scheduler']
    directory, writer, writer_prefix = train_utils['logging']['directory'], train_utils['logging']['writer'], train_utils['logging']['writer_prefix']

    for epoch in tqdm(range(total_epochs,total_epochs+input_args.epochs),desc='Epochs'):
        train_losses = np.zeros(len(train_loader))
        train_branch_losses = np.zeros((len(train_loader),input_args.n_branches))
        train_accuracies = np.zeros((len(train_loader),input_args.n_branches))
        model.train()
        for train_idx, (x, y) in enumerate(train_loader):
            train_step = epoch*len(train_loader) + train_idx
            x,y = x.to(device),y.to(device)
            #1-forward pass - get logits
            output = model(x)

            #2-objective function
            J_train,batch_branch_losses = branched_loss(output,y)
            batch_train_accuracies = get_branched_accuracies(output,y)
            
            #3-clean gradients
            model.zero_grad()
            
            #4-accumulate partial derivatives of J
            J_train.backward()
            
            #5-write metrics to summary file
            train_losses[train_idx] = J_train
            writer.add_scalar(writer_prefix + "Step total loss train",J_train,train_step)
            for idx,branch_loss in enumerate(batch_branch_losses):
                train_branch_losses[train_idx,idx] = branch_loss
                train_accuracies[train_idx,idx] = batch_train_accuracies[idx]
                writer.add_scalar((writer_prefix + "Branch Losses/Branch "+str(idx)),branch_loss,train_step)
                writer.add_scalar((writer_prefix + "Branch Accuracies/Branch  "+str(idx)),batch_train_accuracies[idx],train_step)

            #6-step in opposite direction of gradient
            optimiser.step()

        model.eval()
        val_losses = np.zeros(len(val_loader))
        val_branch_losses = np.zeros((len(val_loader),input_args.n_branches))
        val_accuracies = np.zeros((len(val_loader),input_args.n_branches))
        for val_idx, (x, y) in enumerate(val_loader):
            val_step = epoch*len(val_loader) + val_idx
            x,y = x.to(device),y.to(device)
            #1-forward pass - get logites
            with torch.no_grad():
                output = model(x)

            #2-metrics
            J_val,batch_branch_losses = branched_loss(output,y)
            validation_accuracies_batch = get_branched_accuracies(output,y)
            
            #5-write metrics to summary file
            val_losses[val_idx] = J_val
            
            writer.add_scalar(writer_prefix + "Step total loss train",J_val,val_step)
            for idx,branch in enumerate(batch_branch_losses):
                val_branch_losses[val_idx,idx] = branch_loss
                val_accuracies[val_idx,idx] = validation_accuracies_batch[idx]
                writer.add_scalar((writer_prefix + "Branch validation losses/Branch " +str(idx)),branch,val_step)
                writer.add_scalar((writer_prefix + "Branch validation Accuracies/Branch "+str(idx)),validation_accuracies_batch[idx],val_step)

        epoch_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_loss:
            torch.save(model.state_dict(), directory + '/weights_best.pth')
            best_loss = val_loss

        writer.add_scalar(writer_prefix + "Epoch loss train",epoch_loss,epoch)
        writer.add_scalar(writer_prefix + "Epoch loss val",val_loss,epoch)

        mean_train_branch_losses = train_branch_losses.mean(axis=0)
        mean_val_branch_losses = val_branch_losses.mean(axis=0)

        mean_train_accuracies = train_accuracies.mean(axis=0)
        mean_val_accuracies = val_accuracies.mean(axis=0)

        for idx in range(input_args.n_branches):
            writer.add_scalar((writer_prefix + "Branch epoch losses/Branch "+str(idx)),mean_train_branch_losses[idx],epoch)
            writer.add_scalar((writer_prefix + "Branch epoch accuracies/Branch "+str(idx)),mean_train_accuracies[idx],epoch)

            writer.add_scalar((writer_prefix + "Branch epoch validation losses/Branch "+str(idx)),mean_val_branch_losses[idx],epoch)
            writer.add_scalar((writer_prefix + "Branch validation epoch accuracies/Branch "+str(idx)),mean_val_accuracies[idx],epoch)
        
        scheduler.step(val_loss)
        print('epoch: ', epoch, '\tvalidation loss: ', val_loss, '\taccuracies: ', mean_val_accuracies)

    train_utils['optimisation']['best_loss'] = best_loss
    train_utils['optimisation']['total_epochs'] = total_epochs + input_args.epochs

    if return_best:
        model.load_state_dict(torch.load(directory+'/weights_best.pth'))
    
    return model, train_utils

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']