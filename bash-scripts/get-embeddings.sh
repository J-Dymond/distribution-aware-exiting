#!/bin/bash
cd $HOME/--/ 

source ../Progressive\ Transformers/transformer-env/bin/activate

python main/get_embeddings.py $1 -wf $2 --data CIFAR10 --partition train --batch_size 300
