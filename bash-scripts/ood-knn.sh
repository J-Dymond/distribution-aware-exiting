#!/bin/bash

cd $HOME/--/ 

source ../Progressive\ Transformers/transformer-env/bin/activate

python main/knn_OOD.py $1 --weights_file $2 --data_train CIFAR10 --partition_train train --data_test CIFAR10 --partition_test test --batch_size 128 
python main/knn_OOD.py $1 --weights_file $2 --data_train CIFAR10 --partition_train train --data_test CIFAR100 --partition_test test --batch_size 128 
python main/knn_OOD.py $1 --weights_file $2 --data_train CIFAR10 --partition_train train --data_test SVHN --partition_test test --batch_size 128 
python main/knn_OOD.py $1 --weights_file $2 --data_train CIFAR10 --partition_train train --data_test DTD --partition_test test --batch_size 128 
python main/knn_OOD.py $1 --weights_file $2 --data_train CIFAR10 --partition_train train --data_test tiny-imagenet --partition_test test --batch_size 128 
