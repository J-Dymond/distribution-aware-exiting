from collections import OrderedDict
from operator import sub
import numpy as np

import torch
import torch.nn as nn
import re
import math

class BranchedNet(nn.Module):
    def __init__(self, backbone, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.output_classes = backbone.n_classes
        self.selected_out = OrderedDict()
        self.fhooks = list()
        self.fhook_identifiers = list()

        #module list dynamically creates branch modules
        self.branches = nn.ModuleList()

        #backbone MODEL
        self.backbone = backbone

        branch_selection_function = backbone.branch_func

        #function defined in ResNet code determines the modules name to attach branches to
        self.selected_layers = branch_selection_function(backbone,output_layers)
        
        #Attach forward hooks to pass to branches
        output_filters = list() #list to keep output filters sizes
        for _, (name,module) in enumerate(list(self.backbone.named_modules())):
            if name in self.selected_layers:
                param_shapes = list() #get shapes of the paramaters in the layer
                for _, (_,param) in enumerate(module.named_parameters()):
                    param_shapes.append(param.shape)
                output_filters.append(param_shapes[-1][0]) #final paramater = bias = number of filters
                #attach hooks and keep hook name
                self.fhooks.append(module.register_forward_hook(self.branch_hook(module)))
                self.fhook_identifiers.append(name)

        #create branches for each selected layer
        for idx, _ in enumerate(self.selected_layers):
            #use output filters to create appropriately sized branches
            layer_channels = output_filters[idx]
            self.branches.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(start_dim=1),
                # nn.Dropout(p=0.7),
                nn.Linear((layer_channels),self.output_classes)
                )
            )

    #forward hook will take output from selected layer, put it in dictionary
    def branch_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        outputs = list()
        final_out = self.backbone(x) #run forward pass to obtain output and retrieve forward hooks
        sub_out = self.selected_out.items()

        #pass forward hooks to approriate branch
        for idx, (output) in enumerate(sub_out):
            outputs.append(self.branches[idx](output[1]))

        #final output should be last output
        outputs.append(final_out)
        return outputs


class VariableWidthBranchedNet(nn.Module):
    '''
    This is a BranchedNet module with variable width

    Overview
    -------------------
    
    Model accepts backbone which is split into 'slices' these will be trained independently. 
    The forward method takes an argument width, which determines how many of these slices are required for inference.
    Slices are concatenated together before being passed to each classification layer.
    All slices are the same depth, therefore hook attaching function works the same for each
    '''
    def __init__(self, backbone, output_layers, *args):
        super().__init__(*args)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.output_layers = output_layers
        self.n_branches = output_layers-1
        self.output_classes = backbone.n_classes
        self.selected_out = OrderedDict()
        self.fhooks = list()
        self.fhook_identifiers = list()

        branch_selection_function = backbone.branch_func

        #module list dynamically creates branch modules
        self.branches = nn.ModuleList()

        #backbone MODEL
        self.backbone = backbone

        #function defined in ResNet code determines the modules name to attach branches to
        self.selected_layers = branch_selection_function(backbone.encoder_full,output_layers)

        #Attach forward hooks to pass to branches
        setattr(backbone.encoder_quarter, "name", "encoder_quarter")
        quarter_output_filters = torch.tensor(self.attach_hooks(backbone.encoder_quarter))

        setattr(backbone.encoder_half, "name", "encoder_half")
        half_output_filters = torch.tensor(self.attach_hooks(backbone.encoder_half))

        setattr(backbone.encoder_full, "name", "encoder_full")
        full_output_filters = torch.tensor(self.attach_hooks(backbone.encoder_full))

        self.branch_channels = torch.vstack((quarter_output_filters,half_output_filters,full_output_filters))

        self.create_branches()

    #forward hook will take output from selected layer, put it in dictionary
    def branch_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def set_inference_mode(self,state):
        assert state in ['quarter','half','full'], 'please choose a valid inference mode: quarter, half, full'
        self.inference_mode = state
    
    def set_training_state(self,width):
        #sets the training state for the encoders for a given training method
        if width == 'full':
            self.set_inference_mode('full')
            for _,param in self.backbone.named_parameters():
                param.requires_grad = True
                
        elif width == 'full-fine':
            self.set_inference_mode('full')
            for _,param in self.backbone.encoder_quarter.named_parameters():
                param.requires_grad = False
            for _,param in self.backbone.encoder_half.named_parameters():
                param.requires_grad = False
            for _,param in self.backbone.encoder_full.named_parameters():
                param.requires_grad = True

        elif width == 'half':
            self.set_inference_mode('half')
            for _,param in self.backbone.encoder_quarter.named_parameters():
                param.requires_grad = True
            for _,param in self.backbone.encoder_half.named_parameters():
                param.requires_grad = True
            for _,param in self.backbone.encoder_full.named_parameters():
                param.requires_grad = False
        elif width == 'half-fine':
            self.set_inference_mode('half')
            for _,param in self.backbone.encoder_quarter.named_parameters():
                param.requires_grad = False
            for _,param in self.backbone.encoder_half.named_parameters():
                param.requires_grad = True
            for _,param in self.backbone.encoder_full.named_parameters():
                param.requires_grad = False

        elif width == 'quarter' or width == 'quarter-fine':
            self.set_inference_mode('quarter')
            for _,param in self.backbone.encoder_quarter.named_parameters():
                param.requires_grad = True
            for _,param in self.backbone.encoder_half.named_parameters():
                param.requires_grad = False
            for _,param in self.backbone.encoder_full.named_parameters():
                param.requires_grad = False

    def attach_hooks(self,encoder):
        #given the branch selection function result attach branches to each layer
        output_filters = list() #list to keep output filters sizes
        for _, (name,module) in enumerate(list(encoder.named_modules())):
            if name in self.selected_layers:
                param_shapes = list() #get shapes of the paramaters in the layer
                for _, (_,param) in enumerate(module.named_parameters()):
                    param_shapes.append(param.shape)
                output_filters.append(param_shapes[-1][0]) #final paramater = bias = number of filters
                #attach hooks and keep hook name
                self.fhooks.append(module.register_forward_hook(self.branch_hook(module)))
                self.fhook_identifiers.append(encoder.name+'.'+name)
        return(output_filters)

    def output_select(self,idx):
            #This function calculates which branch and slice the inference is at for a given index
            #(the indexes are flattened to a list)
            branch_idx = idx%(self.n_branches)
            width_idx = math.floor(idx/self.n_branches)
            return(branch_idx,width_idx)


    def create_branches(self):
    #create branches for each selected layer
        for idx, _ in enumerate(self.fhook_identifiers):
            #use output filters to create appropriately sized branches
            #work out which combination of layers is allocated to branch calculate number of channels
            branch_idx, width_idx = self.output_select(idx)
            layer_channels = torch.sum(self.branch_channels[:width_idx+1,branch_idx],dim=-1,dtype=int)

            self.branches.append(
                nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear((layer_channels),self.output_classes)
                )
            )

    def forward(self, x):
        outputs = list()
        #determining inference mode
        if self.inference_mode == 'full':
            channels = 3
        elif self.inference_mode == 'half':
            channels = 2
        elif self.inference_mode == 'quarter':
            channels = 1

        final_out = self.backbone(x,width=self.inference_mode) #run forward pass to obtain output and retrieve forward hooks
        sub_out = list(self.selected_out.items())
        #only recieve the outputs you need for specific inference mode
        branch_inputs = sub_out[:channels*(self.n_branches)]
        #There are different branches for each combination, this allocates the correct ones from module_list
        for idx, branch in enumerate(self.branches[(channels-1)*(self.n_branches):(channels)*(self.n_branches)]):

            #create list of pooled outputs for branch
            pooled_inputs = list()
            for output in branch_inputs[idx:self.n_branches*channels:self.n_branches]:
                pooled_inp = self.adaptive_pool(output[1])
                pooled_inputs.append(pooled_inp)

            #concatenate outputs and pass to branch
            branch_input = torch.cat(pooled_inputs,dim=1)
            outputs.append(branch(branch_input))

        #final output should be last output, the specific channel combination for this is determined in the forward method of: backbone(x)
        outputs.append(final_out)
        return outputs