# SPEID

<h1>Overview</h1>

SPEID is a deep neural network (implemented in Python, based on the Theano and Keras libraries) designed for predicting enhancer-promoter interactions directly from sequence data, as described in 

Singh, Shashank, et al. "Predicting Enhancer-Promoter Interaction from Genomic Sequence with Deep Neural Networks." bioRxiv (2016): 085241. http://biorxiv.org/content/early/2016/11/02/085241.

<h1>Data</h1>
The main data requirement for training SPEID is DNA sequences of the positive and negative enhancer promoter pairs in the desired cell type. For convenient reproducability in the cell lines we studied (GM12878, HeLa-S3, HUVEC, IMR90, K562, and NHEK) we have packaged these data together in a single HDF5 file available at TODO.

One can optionally initialize a number of the convolution kernels with motifs from the JASPAR database. These motifs can be conveniently accessed from the numpy file available at https://github.com/uci-cbcl/DanQ/blob/master/JASPAR_CORE_2016_vertebrates.npy.

<h1>SPEID Training/Prediction</h1>

Running basic_training.py will train a model for each cell line and save the learned weights. Note that training SPEID is prohibitively slow without a GPU, and, even with a GPU, can take between a few hours and a few days (per cell line), depending on the GPU. On our setup (using a single NVIDIA GTX 1080), SPEID takes about 20 hours to train (per cell line).

If you are only interested in using SPEID for prediction, you can download our learned weights from TODO and run [TODO: Add simple file for runnning predictions.]. Note that prediction is much faster than training, taking a few minutes with a decent GPU, and a few hours otherwise.

<h1>Estimating Feature Importance with SPEID</h1>

TODO
