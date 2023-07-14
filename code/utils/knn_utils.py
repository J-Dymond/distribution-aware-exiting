import os
import random
import sys

import numpy as np 
import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer, required

from torchvision import datasets, transforms

class get_penultimate_input(nn.Module):
    def __init__(self,model,preprocessing_module,final_activation_hook,return_all=False,return_model_out=False):
        super().__init__()
        assert type(return_all) == bool , 'return all should be True or False'
        assert type(return_model_out) == bool , 'return_model_out should be True or False'
        self.model = model
        self.preprocessing_module = preprocessing_module
        self.final_activation_hook = final_activation_hook   
        self.return_all = return_all
        self.return_model_out = return_model_out
                       
    def forward(self,inp,branch_idx):

        if self.return_model_out:
            model_out = self.model(inp)
        else:
            _ = self.model(inp)

        branches = list(self.model.selected_out.items())
        if self.return_all == False:
            if self.return_model_out:
                #avoids inplace operation if it might cause issue
                placeholder_out = model_out[branch_idx]
                model_out = placeholder_out
            if branch_idx < len(branches):
                new_input = self.preprocessing_module(branches[branch_idx][1])
                out = new_input 
            else:
                new_input = self.preprocessing_module(self.final_activation_hook['encoder_output'])
                out = new_input    
            
        elif self.return_all == True:
            out = list()
            for branch_idx,branch_out in enumerate(branches):
                if branch_idx < len(branches):
                    new_input = self.preprocessing_module(branches[branch_idx][1])
                    out.append(new_input)
                    
            new_input = self.preprocessing_module(self.final_activation_hook['encoder_output'])
            out.append(new_input)
        
        if self.return_model_out:
            return out,model_out
        else:
            return out