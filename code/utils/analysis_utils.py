import numpy as np 
import torch
import torch.nn as nn
import os
import re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def entropy(prediction):
    log = prediction*np.log(prediction)
    ent = -np.sum(log)
    return ent


def get_distribution(outputs,labels,powers,policy='entropic',n_thresh=25):
    n_inputs = outputs.shape[0]
    n_branches = outputs.shape[1]

    if policy == 'entropic':
        predictions,power_usage,exits = entropic_distributions(outputs,powers,n_inputs,n_branches,n_thresh)
    
    accuracy_distribution = get_accuracy_metrics(predictions,labels)
    power_distribution = (power_usage/n_inputs)

    exit_p = np.mean(exits,axis=1)
    

    return accuracy_distribution,power_distribution,exit_p

def get_distribution_with_exits(outputs,labels,powers,exit_probs,n_thresh=25):
    n_inputs = outputs.shape[0]
    n_branches = outputs.shape[1]
        
    predictions,power_usage,exits = get_exits(outputs,powers,n_inputs,n_branches,n_thresh,exit_probs)
    
    accuracy_distribution = get_accuracy_metrics(predictions,labels)
    power_distribution = (power_usage/n_inputs)

    exit_p = np.mean(exits,axis=1)
    
    
    return accuracy_distribution,power_distribution,exit_p

def get_OOD_usage(ood_entropies,policy='entropic',n_thresh=25):
    n_branches = ood_entropies.shape[1]
    accs = np.zeros((n_thresh,n_branches))
    thresholds = np.linspace(0,1,n_thresh)
    for thresh_idx,threshold in enumerate(thresholds):
        accs[thresh_idx,:] = np.sum(ood_entropies < threshold,dtype=int,axis=0)/len(ood_entropies)
    return accs

def get_exits(outputs,powers,n_inputs,n_branches,n_thresh,probablilties):
    branch_predictions = np.argmax(outputs,axis=2)
    
    exits = np.zeros((n_thresh,n_inputs,n_branches))
    predictions = np.zeros((n_thresh,n_inputs))
    power_usage = np.zeros(n_thresh)

    thresholds = np.linspace(0,1,n_thresh)
    for thresh_idx,threshold in enumerate(thresholds):
        for inp_idx in range(n_inputs):
            early_exit = False
            for brch_idx,branch_prob in enumerate(probablilties[inp_idx,:]):
                if branch_prob > threshold:
                    exits[thresh_idx,inp_idx,brch_idx] = 1
                    predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,brch_idx] 
                    early_exit=True
                    break
            if early_exit == False:
                exits[thresh_idx,inp_idx,(n_branches-1)] = 1
                predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,(n_branches-1)]
        power_usage[thresh_idx] = np.dot(np.sum(exits[thresh_idx,:],axis=0),powers)
    
    return predictions, power_usage, exits

def entropic_distributions(outputs,powers,n_inputs,n_branches,n_thresh):

    branch_predictions = np.argmax(outputs,axis=2)

    entropies = np.zeros((n_inputs,n_branches))
    for input_idx in range(n_inputs):
        for branch_idx in range(n_branches):
            entropies[input_idx,branch_idx] = entropy(outputs[input_idx,branch_idx,:])
    
    exits = np.zeros((n_thresh,n_inputs,n_branches))
    predictions = np.zeros((n_thresh,n_inputs))
    power_usage = np.zeros(n_thresh)

    max_entropy = np.log(outputs.shape[2])  
    thresholds = np.linspace(max_entropy,0,n_thresh)
    for thresh_idx,threshold in enumerate(thresholds):
        for inp_idx in range(n_inputs):
            early_exit = False
            for brch_idx,branch_entropy in enumerate(entropies[inp_idx,:]):
                if branch_entropy < threshold:
                    exits[thresh_idx,inp_idx,brch_idx] = 1
                    predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,brch_idx] 
                    early_exit=True
                    break
            if early_exit == False:
                exits[thresh_idx,inp_idx,(n_branches-1)] = 1
                predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,(n_branches-1)]
        power_usage[thresh_idx] = np.dot(np.sum(exits[thresh_idx,:],axis=0),powers)
    
    return predictions, power_usage, exits

def get_optimum_exits(outputs,labels):
    predictions = np.argmax(outputs,axis=2)
    correct = list()

    n_branches = predictions.shape[1]+1
    n_inputs = predictions.shape[0]+1

    for branch in range(n_branches-1):
        correct.append(np.equal(predictions[:,branch],labels,dtype=int))
    #to quickly get it to the correct shape (matching predictions array)

    all_correct = np.array(np.dstack(correct)[0],dtype=int)
    optimum_exits = np.zeros(((n_inputs-1),(n_branches-1)))

    for label_idx,label in enumerate(labels):
        for branch_idx,branch_prediction in enumerate(predictions[label_idx,:]):
            if branch_prediction == label:
                optimum_exits[label_idx,branch_idx] = 1
                break

    return(all_correct,optimum_exits)


def get_accuracy_metrics(predictions,labels,sigma=3):
    n_thresh = predictions.shape[0]
    n_inputs = predictions.shape[1]
    accuracy = np.zeros((n_thresh,2))
    for thresh_idx in range(n_thresh):
        accuracies = (predictions[thresh_idx,:] == labels)
        mean_acc = np.mean(accuracies)
        interval = sigma * np.sqrt((mean_acc*(1-mean_acc))/n_inputs)
        accuracy[thresh_idx,0] = np.mean(accuracies)
        accuracy[thresh_idx,1] = interval

    return(accuracy)


def load_values(directory,callibration = False):
    values = dict()

    metric_directory = directory+'/analysis/'

    values['ground_truth'] = np.load(metric_directory+'labels.npy')
    values['output'] = np.load(metric_directory+'outputs.npy')
    values['raw_output'] = np.load(metric_directory+'logit_outputs.npy')
    
    if callibration:
        callibration_directory = directory+'/analysis/callibration/'
        values['train_ground_truth'] = np.load(callibration_directory+'train_labels.npy')
        values['train_output'] = np.load(callibration_directory+'train_outputs.npy')
        values['train_raw_output'] = np.load(callibration_directory+'train_logit_outputs.npy')

        values['val_ground_truth'] = np.load(callibration_directory+'val_labels.npy')
        values['val_output'] = np.load(callibration_directory+'val_outputs.npy')
        values['val_raw_output'] = np.load(callibration_directory+'val_logit_outputs.npy')

    return values

class Optimum_Exit(Dataset):
    def __init__(self, data, targets):
        #data:      N_inputs x N_classes x N_branches
        #targets:   N_inputs x N_branches
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def create_decision_module(model,device,pretrained=None):
    """
    Creates a Decision module
    Inputs: Model, Device to load onto, pretrained_weights (if using, defaults to none)
    Returns: Decision modules, preprocessing module, with loaded weights if using
    """
    Decision_Modules = nn.ModuleList()
    if not pretrained:
        best_weights = list()

    #only need decision module for the branches
    for branch_idx,branch in enumerate(model.branches):
        input_features = branch[-1].in_features
        for _,prev_branch in enumerate(model.branches[:branch_idx]):
            input_features += prev_branch[-1].in_features
            print(input_features)
            #each branch will get all of the inputs before it, as well as the input for the branch
        Decision_Modules.append(
            nn.Sequential(
                nn.Linear((input_features),1),
                nn.Flatten(start_dim=0),
                nn.Sigmoid())
            )
        if not pretrained:
            best_weights.append(Decision_Modules[branch_idx].state_dict())

    #this preprocess the branch input so it can be stacked efficiently, it has no trainable parameters
    preprocessing_module = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(start_dim=1)).to(device)

    if pretrained is None:
        Decision_Modules.to(device)
        preprocessing_module = preprocessing_module.to(device)
        return Decision_Modules,best_weights,preprocessing_module
    else:
        Decision_Modules.load_state_dict(torch.load(pretrained,map_location=device))
        Decision_Modules = Decision_Modules.to(device)
        preprocessing_module = preprocessing_module.to(device)
        return Decision_Modules,preprocessing_module

def get_optimum_policy(directory):
    outputs = np.load(directory+'/outputs.npy')
    labels = np.load(directory+'/labels.npy')
    powers = np.load(directory+'/power_usage.npy')

    predictions = np.argmax(outputs,axis=2)
    correct = list()

    n_branches = predictions.shape[1]+1
    n_inputs = predictions.shape[0]+1

    for branch in range(n_branches-1):
        correct.append(np.equal(predictions[:,branch],labels,dtype=int))
    
    #to quickly get it to the correct shape (matching predictions array)
    all_correct = np.dstack(correct)[0]
    print(all_correct.shape)
    print(predictions.shape)
    print(labels.shape)
    optimum_exits = np.zeros(((n_inputs-1),(n_branches-1)))
    correct = np.zeros((n_inputs-1))

    for label_idx,label in enumerate(labels):
        for branch_idx,branch_prediction in enumerate(predictions[label_idx,:]):
            if branch_prediction == label:
                optimum_exits[label_idx,branch_idx] = 1
                correct[label_idx] = 1
                break

    return(optimum_exits,correct,powers)

def set_unique_exp_name(save_directory,exp_name):
    try:
        directory = save_directory+exp_name+'/'
        os.mkdir(directory)
        print("Saving data to: " , directory)
        return(directory)

    except FileExistsError:
        print(directory, "Already exists...")
        for run in range(1,100):
            try:
                directory = save_directory+exp_name+'_'+str(run)+"/"
                os.mkdir(directory)
                print("Instead saving data to: " , directory)
                return(directory)

            except FileExistsError:
                continue

def stack_embeddings(model_directory,dataset,partition):
    
    embeddings_directory = model_directory+'/embeddings/'+dataset+'/'+partition+'/'
    
    branches = os.listdir(embeddings_directory)
    if '.DS_Store' in branches:
        branches.remove('.DS_Store')
    n_branches = len(branches)


    for branch_idx in range(n_branches):
        embedding_stack = list()
        branch_directory = embeddings_directory + '/branch_' + str(branch_idx) + '/'
        for file in os.listdir(branch_directory):
            #this ensures that the all_embeddings.pt file is not loaded by accident
            if re.search(r'batch_\d+', file):
                batch = torch.load(branch_directory+file)
                embedding_stack.append(batch)
                del batch
        all_embeddings = torch.vstack(embedding_stack)
        torch.save(all_embeddings,branch_directory+'all_embeddings.pt')
        del embedding_stack
        del all_embeddings

def load_embeddings(model_directory,dataset,partition):
    embeddings_directory = model_directory+'/embeddings/'+dataset+'/'+partition+'/'
    branches = os.listdir(embeddings_directory)
    if '.DS_Store' in branches:
        branches.remove('.DS_Store')
    n_branches = len(branches)
    all_embeddings = list()
    for branch_idx in range(n_branches):
        branch_directory = embeddings_directory + '/branch_' + str(branch_idx) + '/'
        branch_embeddings = torch.load(branch_directory+'all_embeddings.pt')
        all_embeddings.append(branch_embeddings)
        del branch_embeddings
    return all_embeddings

def plot_macs_vs_acc(directory,exit_policy='entropic', exits_provided = False, label_note = '', decision_module_path = None, colour=False, label=True):

    name = directory.split('/')[3] + ' -- ' + directory.split('/')[4]
    # name = directory.split('/')[4]
    name = name.replace('_',' ') + label_note

    # name = directory.split('/')[4].split('d')[-1] + ' layer'

    outputs = np.load(directory+'/outputs.npy')
    labels = np.load(directory+'/labels.npy')
    powers = np.load(directory+'/power_usage.npy')
    if exits_provided == True:
        if exit_policy == 'entropic':
            exit_probs = 1-np.load(directory+'/exits/entropic.npy')
        elif exit_policy == 'decision_module':
            assert decision_module_path, 'please specify which decision module to examine'
            exit_probs = np.load(directory+'/exits/decision_modules/'+decision_module_path+'.npy')
        else:
            print('Invalid exit policy selected: Using entropic')
            exit_probs = 1-np.load(directory+'/exits/entropic.npy')

        accs,macs,exit_p = get_distribution_with_exits(outputs,labels,powers,exit_probs,n_thresh=25)
    else:
        accs,macs,exit_p = get_distribution(outputs,labels,powers,policy=exit_policy)

    mean_accs = accs[:,0]
    std_accs = accs[:,1]

    if label and colour:
        plt.plot(macs,mean_accs,label=name,color=colour)
        plt.fill_between(macs,mean_accs+std_accs,mean_accs-std_accs,alpha=0.2,color=colour)
    elif label and not colour:
        plt.plot(macs,mean_accs,label=name)
        plt.fill_between(macs,mean_accs+std_accs,mean_accs-std_accs,alpha=0.2)
    elif colour and not label:
        plt.plot(macs,mean_accs,color=colour)
        plt.fill_between(macs,mean_accs+std_accs,mean_accs-std_accs,alpha=0.2,color=colour)
    else:
        plt.plot(macs,mean_accs)
        plt.fill_between(macs,mean_accs+std_accs,mean_accs-std_accs,alpha=0.2)
        
    if exits_provided == True:
        return exit_probs
    

def run_ood_inference(model_directory,train_dataset,ood_dataset,n_thresh=100,knn_percentile=0.95,detect_ood=True):
    func_outputs = dict()
    name = model_directory.split('/')[3] + ' -- ' + model_directory.split('/')[4]

    id_outputs = np.load(model_directory+'/outputs.npy')
    n_branches = id_outputs.shape[1]
    id_labels = np.load(model_directory+'/labels.npy')

    powers = np.load(model_directory+'/power_usage.npy')

    id_knn = np.load(model_directory+'/../knn_ood/train_'+train_dataset+'_train/test_'+train_dataset+'_test/all_k_distances.npy')
    ood_knn = np.load(model_directory+'/../knn_ood/train_'+train_dataset+'_train/test_'+ood_dataset+'_test/all_k_distances.npy')

    branch_predictions = np.argmax(id_outputs,axis=2)

    n_id_inputs = branch_predictions.shape[0]
    n_ood_inputs = ood_knn.shape[0]

    ood_labels = np.full(n_ood_inputs,-1)
    all_labels = np.concatenate([id_labels,ood_labels])

    id_entropies = np.zeros((n_id_inputs,n_branches))
    for input_idx in range(n_id_inputs):
        for branch_idx in range(n_branches):
            id_entropies[input_idx,branch_idx] = entropy(id_outputs[input_idx,branch_idx,:])
    
    id_exits = np.zeros((n_thresh,n_id_inputs,n_branches))
    id_predictions = np.zeros((n_thresh,n_id_inputs))

    ood_exits = np.zeros((n_thresh,n_ood_inputs,n_branches))
    ood_predictions = np.zeros((n_thresh,n_ood_inputs))

    power_usage = np.zeros(n_thresh)

    max_entropy = np.log(id_outputs.shape[2])  
    thresholds = np.linspace(max_entropy,0,n_thresh)

    knn_thresholds = np.zeros(n_branches)
    for branch in range(n_branches):
        id_branch = id_knn[:,branch,-1]
        knn_thresholds[branch] = np.percentile(id_branch,knn_percentile*100)
    
    for thresh_idx,threshold in enumerate(thresholds):
        #get ID outputs w/ knn 
        for inp_idx in range(n_id_inputs):
            early_exit = False
            for branch_idx,branch_entropy in enumerate(id_entropies[inp_idx,:]):
                if id_knn[inp_idx,branch_idx,-1] > knn_thresholds[branch_idx]:
                    id_exits[thresh_idx,inp_idx,branch_idx] = 1
                    id_predictions[thresh_idx,inp_idx] = -1
                    early_exit=True
                    break
                if branch_entropy < threshold:
                    id_exits[thresh_idx,inp_idx,branch_idx] = 1
                    id_predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,branch_idx] 
                    early_exit=True
                    break
            if early_exit == False:
                id_exits[thresh_idx,inp_idx,(n_branches-1)] = 1
                id_predictions[thresh_idx,inp_idx] = branch_predictions[inp_idx,(n_branches-1)]

        #get OOD knn outputs w/ entropy
        for inp_idx in range(n_ood_inputs):
            if not detect_ood:
                ood_exits[:,:,(n_branches-1)] = 1
                ood_predictions[:,:] = -2
                break
            early_exit = False
            for branch_idx,knn_distance in enumerate(ood_knn[inp_idx,:,-1]):
                if knn_distance > knn_thresholds[branch_idx]:
                    ood_exits[thresh_idx,inp_idx,branch_idx] = 1
                    ood_predictions[thresh_idx,inp_idx] = -1
                    early_exit=True
                    break
            if early_exit == False:
                ood_exits[thresh_idx,inp_idx,(n_branches-1)] = 1
                ood_predictions[thresh_idx,inp_idx] = -2

        all_exits = np.concatenate([id_exits,ood_exits],axis=1) 
        power_usage[thresh_idx] = np.dot(np.sum(all_exits[thresh_idx,:],axis=0),powers)
        all_predictions = np.concatenate([id_predictions,ood_predictions],axis=1) 

    ood_accuracy = np.zeros(n_thresh)
    id_accuracy = np.zeros(n_thresh)
    all_accuracy = np.zeros(n_thresh)
    for thresh in range(n_thresh):
        id_accuracy[thresh] = np.mean(id_predictions[thresh,:] == id_labels) 
        ood_accuracy[thresh] = np.mean(ood_predictions[thresh,:] == ood_labels) 
        all_accuracy[thresh] = np.mean(all_predictions[thresh,:] == all_labels) 
    
    func_outputs['all_predictions'] = all_predictions
    func_outputs['power_usage'] = power_usage/len(all_labels)
    func_outputs['all_labels'] = all_labels
    func_outputs['all_exits'] = all_exits
    func_outputs['id_predictions'] = id_predictions
    func_outputs['id_labels'] = id_labels
    func_outputs['ood_labels'] = ood_labels
    func_outputs['id_accuracy'] = id_accuracy
    func_outputs['ood_accuracy'] = ood_accuracy
    func_outputs['all_accuracy'] = all_accuracy


    return func_outputs
