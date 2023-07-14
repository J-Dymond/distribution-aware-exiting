#!/bin/bash
cd $HOME/--/

source ../Progressive\ Transformers/transformer-env/bin/activate

python main/train.py $1 --backbone resnet --depth 50 --width 1.0 -nb 4 -ds CIFAR10 -ep 1 -lr 0.01 -l cross-entropy -rd True
