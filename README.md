## Exploiting Epistemic Uncertainty at Inference Time 

This repository contains the code used for the Paper titled [Exploiting epistemic uncertainty at inference time for early-exit power saving](). 

It also contains some of the code used in the paper [Adapting Branched Networks to Enable Progressive Intelligence](https://bmvc2022.mpi-inf.mpg.de/0990.pdf), namely in the folder <tt>main/models</tt>. 

The repository is organised as such:

- <tt>main</tt>
    - sub directories:
        - <tt>models/</tt>: 
            - code for producing ResNet models of any depth/width, and attaching any number of branches to them. There is also code for producing 'slimmable' models, with 3 operating widths at inference time.
        - <tt>notebooks/</tt>:
            - Analysis notebooks 
        - <tt>utils/</tt>:
            - Utility files for calling models, dataset, etc.
        - <tt>ptflops/</tt>: 
            - Adapted code from [this repository](https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/flops_counter.py) used for calculating model parameter counts and MAC usage.
    - scripts:
        - <tt>train.py</tt>
            - Code for training the branched models
        - <tt>slimmable_train.py</tt>
            - Code for training 'slimmable' branched models
        - <tt>get_embeddings.py</tt>
            - Code for retrieving the embeddings on the train/validation set
        - <tt>knn_OOD.py</tt>
            - This performs the knn measurements using the OOD test sets
        - <tt>get_predictions.py</tt>
            - This performs the inference on the test set, to obtain the accuracy on each ID set, it also saves the raw outputs
- <tt>bash-scripts</tt>
    - This contains bash scripts for running experiments using <tt>SLURM</tt> workload manager
    - These can be called to run the experiments of the paper
