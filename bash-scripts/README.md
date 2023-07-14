# Bash scripts for <tt>SLURM</tt> workload manager 
### This folder contains some bash scripts that are used to run the scripts 

##### These scripts will create a model, you need to specify the name of the model with command line input

- <tt>train.sh</tt> - runs training script, trains model of desired size on selected dataset, saves it accordingly
- <tt>train-slimmable.sh</tt> - runs training script with slimmable model, trains model of desired size on selected dataset, saves it accordingly

##### These need to have a specified directory to load the model from, you can specify this with command line input, you also pass the dataset in via command line, in the second position

- <tt>run-analysis.sh</tt> - runs evaluation script, saves the accuracy, loss, and raw outputs on test set data
- <tt>run-analysis-slimmable.sh</tt> - runs evaluation script with slimmable model, saves the accuracy, loss, and raw outputs on test set data

##### These scripts also need the specified directory, but it supports additional weight files, hence these scripts also need to have the weights-file detailed in command line. This is relative to the specified experiment directory, in the first position. In get-embeddings the dataset is hardcoded as CIFAR10 here, but if trained on a different dataset it should be set to that dataset. 

- <tt>get-embeddings.sh</tt> - runs get_embeddings script, returns the embeddings on the selected data (defaults to the train set)
- <tt>ood-knn.sh</tt> - runs knn_OOD script, uses embeddings to calculate the knn_ood
