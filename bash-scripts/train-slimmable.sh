#!/bin/bash

python code/slimmable_train.py $1 --backbone resnet --depth 18 --width 1.0 -nb 4 -ds CIFAR10 -lr 0.5 -rd True