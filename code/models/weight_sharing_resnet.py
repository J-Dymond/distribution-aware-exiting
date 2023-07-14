from numpy import Inf
import torch
import torch.nn as nn
import numpy as np
import re

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], width_multiplier=1., depths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, resolution=224, *args,**kwargs):
        super().__init__()
        
        for block_idx in range(len(blocks_sizes)):
            blocks_sizes[block_idx] = int(blocks_sizes[block_idx]*width_multiplier)
        
        self.blocks_sizes = blocks_sizes
        
        if resolution > 64:

            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=3, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.blocks_sizes[0]),
                activation(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # can make some changes to make model work more effectively on CIFAR10/100 data
        else:
                self.gate = nn.Sequential(
                nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.blocks_sizes[0]),
            )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class  ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

def assign_blocks(depth,basic_block=False):
    '''
    This assigns the block type and block counts based on the depth of the model

    Parameters
    -------------------
    depth: depth of the ResNet

    Outputs
    -------------------
    block_depth: The depth of the blocks that the ResNet should use
    block_type: The type of block the ResNet should use
    
    '''
    assert(depth>3)
    block_depths = [0,0,0,0]
    block_layers = depth - 2

    if depth > 34:
        block_size = 3
        block_type = ResNetBottleNeckBlock
    else:
        block_size = 2
        block_type = ResNetBasicBlock
    
    if basic_block == True:
        block_size = 2
        block_type = ResNetBasicBlock

    if depth <= 101:
        max_block_depths = [3,4,Inf,3]
    else:
        max_block_depths = [3,8,Inf,3]

    n_blocks = int(block_layers/block_size)
    for block_n in range(1,n_blocks+1):
        block_idx = block_n%4 
        allocated = False
        while not allocated:
            block_count = block_depths[block_idx-1]
            if block_count < max_block_depths[block_idx-1]:
                allocated = True
                block_depths[block_idx-1] += 1
            else:
                block_idx += 1
                block_idx = block_idx%4

    return(block_depths,block_type)

def branch_points(resnet,n_branches):
    '''
    This assigns the branch connection points of a branched resnet

    Parameters
    -------------------
    n_branches: number of branches to connect
    resnet: the resnet model itself

    Outputs
    -------------------
    branch_connections: name of the torch modules to connect the branches to
    '''
    possible_connections = list()
    for _, (name,_) in enumerate(list(resnet.named_modules())):
        if re.search(r"^((?!shortcut).)*conv$", name):
            possible_connections.append(name)

    selected_layers = np.linspace(0,len(possible_connections)-1,n_branches+1).astype(int)[1:-1]

    branch_connections = np.array(possible_connections)[selected_layers]
    return branch_connections

class VariableWidthResNet(nn.Module):
    '''
    This is a ResNet module with variable width

    Overview
    -------------------
    
    Model is split into 3 'slices' these will be trained independently. 
    The forward method takes an argument width, which determines how many of these slices are required for inference.
    Slices are concatenated together before being passed to a classification layer.
    '''
    def __init__(self, in_channels, n_classes, width=1.0, *args, **kwargs):
        super().__init__()
        self.inputs = locals() 

        self.encoder_quarter = ResNetEncoder(in_channels, blocks_sizes=[64, 128, 256, 512], width_multiplier = 0.25*width, **kwargs)
        self.encoder_half = ResNetEncoder(in_channels, blocks_sizes=[64, 128, 256, 512], width_multiplier = 0.25*width ,**kwargs)
        self.encoder_full = ResNetEncoder(in_channels, blocks_sizes=[64, 128, 256, 512], width_multiplier = 0.5*width, **kwargs)

        quarter_channels = self.encoder_quarter.blocks[-1].blocks[-1].expanded_channels
        half_channels = self.encoder_quarter.blocks[-1].blocks[-1].expanded_channels + self.encoder_half.blocks[-1].blocks[-1].expanded_channels
        full_channels = self.encoder_quarter.blocks[-1].blocks[-1].expanded_channels + self.encoder_half.blocks[-1].blocks[-1].expanded_channels +self.encoder_full.blocks[-1].blocks[-1].expanded_channels
        
        self.decoder_quarter = ResnetDecoder(quarter_channels, n_classes)
        self.decoder_half = ResnetDecoder(half_channels, n_classes)
        self.decoder_full = ResnetDecoder(full_channels, n_classes)

        self.branch_func = branch_points
        self.n_classes = n_classes
        
    def forward(self, x, width='full'):
        x_quarter = self.encoder_quarter(x)
        x_half = self.encoder_half(x)
        x_full = self.encoder_full(x)
        
        #depending on inference mode, calculate correct x for final layer
        if width == 'full':
            x = torch.cat((x_quarter,x_half,x_full),dim=1)
            x = self.decoder_full(x)
        elif width == 'half':
            x = torch.cat((x_quarter,x_half),dim=1)
            x = self.decoder_half(x)
        elif width == "quarter":
            x = x_quarter
            x = self.decoder_quarter(x)
        return x