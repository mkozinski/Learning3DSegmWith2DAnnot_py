# prerequisites
1. python 3
1. torch 0.4.1

# Learning 3D Segmentation With 2D Annotations
This is an implementation of the method described in the paper  
"[Learning to Segment 3D Linear Structures Using Only 2D Annotations](https://infoscience.epfl.ch/record/256857)".  
It contains a demonstration limited to the publicly available MRA dataset, referred to in the paper as "Angiography".

Create a new folder
Clone general network training routines, the code for preprocessing of the dataset, and the experiment code into that folder 
`git clone https://github.com/mkozinski/NetworkTraining_py`  
`git clone https://github.com/mkozinski/MRAdata_py`  
`git clone https://github.com/mkozinski/Learning3DSegmWith2DAnnot_py`  

To fetch and preprocesss the dataset  
`cd MRAdata`  
`./prepareData.sh`

To run baseline training on 3D annotations  
`cd Learning3DSegmWith2DAnnot`  
`python "run_baseline_v1.py"`  
The training progress is logged to a directory called `log_baseline_v1`.

To run training on 2D annotations  
`cd Learning3DSegmWith2DAnnot`  
`python "run_3projections_v1.py"`   
The log directory for this training is `log_3projections_v1`.

In both cases the training progress can be plotted in gnuplot:  
a) the epoch-averaged loss on the training data `plot "<logdir>/logBasic.txt" u 1`,  
b) the F1 performance on the test data `plot "<logdir>/logF1Test.txt" u 1`.

The training loss and test performance plots are not synchronised as testing is performed once every 50 epochs.
Moreover, the logs generated for the two experiments are synchronised in terms of the number of epochs, but not in terms of the number of updates.

The networks weights are dumped in the log directories:  
a) the recent network is stored at `<logdir>/net_last.pth`,  
b) the network attaining the highest F1 score on the test set is stored at `<logdir>/net_Test_bestF1.pth`.

To generate prediction for the test set using a trained network run  
`python "segmentTestSet.py"`.  
The name of the file containing the network, and the output directory are defined at the beginning of the script.
The output is saved in form of stacks of png images.

[This](http://documents.epfl.ch/users/k/ko/kozinski/www/brain_vasculature.gif) is example output for a test volume held out from training, after around 100 000 updates, for a network scoring 76% IoU on the whole held out test set.  
The segmentation has been visualized with [slicer](https://www.slicer.org/), in particular using the "Volume Rendering" and "Screen Capture" modules.
