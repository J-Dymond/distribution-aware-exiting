## Models

This code is used in the paper [Adapting Branched Networks to Enable Progressive Intelligence](https://bmvc2022.mpi-inf.mpg.de/0990.pdf), to create branched models using a ResNet backbone. The ResNet backbone script is dynamic depending on the depth and width of the desired model. Branched model will create a wrapper which attaches the branches to a model using a forward hook, allowing us to simulate a branched model of any width, depth, and number of branches.

This code is used in the file <tt>main/utils/get_model.py</tt>, but basic usage is as such: 

The model can be created like so: 

     backbone = ResNet(in_channels, n_classes, blocks_sizes=[64, 128, 256, 512], block=block_type, depths=blocks, width_multiplier = width, resolution = res)

Where <tt>blocks</tt> and <tt>block_type</tt> are assigned using the function call below, you can also select basic block or bottleneck block as a bool: 

     blocks,block_type = assign_blocks(depth,basic_block = basic:bool)


If <tt>basic_block</tt> is left as False, the resnet function will autamatically select the appropriate block type based on the selected depth. This process can be automated using the function <tt>create_model()</tt> in <tt>main/utils/get_model.py</tt>.

To create a branched model, you can run <tt>attach_branches</tt> found in <tt>main/utils/get_model.py</tt>

    model = attach_branches(input_params,backbone,n_branches)

This will return a branched model, using the created backbone and with <tt>n_branches</tt>. These functions all work on 'slimmable' networks, as well as standard networks. Slimmable networks have 3 inference modes, of varying widths. 