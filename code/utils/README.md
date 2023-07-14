# Utils

## <tt>get_data.py</tt>
This module provides functions to load and preprocess datasets for training and testing machine learning models. The main function is: 
##### <tt>get_dataset(dataset_name, batch_size, shuffling=True, val_set=True, train_augmentations='standard', two_crop=False)</tt>
- Load and preprocess the specified dataset.

- Arguments:

    - dataset_name (str): Name of the dataset.
    - batch_size (int): Batch size for the data loaders.
    - shuffling (bool): Whether to shuffle the data in the data loaders. Default is True.
    - val_set (bool): Whether to create a validation set from the training data. Default is True.
    - train_augmentations (str): Type of augmentations to apply to the training data. Default is 'standard'.
    - two_crop (bool): Whether to create two crops of the same image for training. Default is False.

- Returns:
    - If val_set is True:
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        - data_props (dict): Dictionary containing properties of the dataset.
    - If val_set is False:
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        - data_props (dict): Dictionary containing properties of the dataset.

## <tt>get_model.py</tt>

This module provides code to create a model of any size, and attach the desired number of branches.

##### <tt>create_model(model_hyperparameters)</tt>
- Function to create a model given a hyperparameters dictionary.

- Inputs:

    - model_hyperparameters (dict): A dictionary containing the following keys:
        - model_name (str): The name of the model type to be loaded.
        - depth (int): The depth of the model.
        - width (float): The width multiplier on the model.
        - in_channels (int): The number of input channels to the model.
        - n_classes (int): The number of classes for classification.
        - res (int): The resolution of the model.
- Outputs:
    - model (torch.nn.Module): A PyTorch model class.

##### attach_branches(model_hyperparameters, model, n_branches)
- Function to attach branches to a given model.

- Inputs:
    - model_hyperparameters (dict): A dictionary containing the model hyperparameters.
    - model (torch.nn.Module): The base model to attach branches to.
    - n_branches (int): The number of branches to attach.
- Outputs:
    - Returns a branched version of the input model with the specified number of branches.

 The ResNet class has a built in function, <tt>branch_points()</tt>, which determines where to connect the branches, found in <tt>/main/models/resnet.py</tt>>:

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
