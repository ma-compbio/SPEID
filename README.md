# SPEID

<h1>Overview</h1>

SPEID is a deep neural network (implemented in Python, based on the Theano and Keras libraries) designed for predicting enhancer-promoter interactions (EPIs) directly from sequence data, as described in 

Singh, Shashank, Yang, Yang, Poczos, Barnabas, and Ma, Jian. "Predicting Enhancer-Promoter Interaction from Genomic Sequence with Deep Neural Networks." bioRxiv (2016): 085241. http://biorxiv.org/content/early/2016/11/02/085241.

<h1>Data</h1>
The main data requirement for training SPEID is DNA sequences of the positive and negative enhancer promoter pairs in the desired cell type. For convenient reproducability in the cell lines we studied (GM12878, HeLa-S3, HUVEC, IMR90, K562, and NHEK) we have packaged these data together in a single HDF5 file available at http://genome.compbio.cs.cmu.edu/~sss1/SPEID/all_sequence_data.h5 (note that this is a large-ish (31GB) file).

One can optionally initialize a number of the convolution kernels with motifs from the JASPAR database. These motifs can be conveniently accessed from the numpy file available at https://github.com/uci-cbcl/DanQ/blob/master/JASPAR_CORE_2016_vertebrates.npy.

<h1>SPEID Training/Prediction</h1>

Running basic_training.py will train a model for each cell line and save the learned weights. Note that training SPEID is prohibitively slow without a GPU, and, even with a GPU, can take between a few hours and a few days (per cell line), depending on the GPU. On our setup (using a single NVIDIA GTX 1080), SPEID takes about 20 hours (per cell line) to train.

If you are only interested in using SPEID for prediction, you can download our learned weights from http://genome.compbio.cs.cmu.edu/~sss1/SPEID/ and run [TODO: Add simple file for runnning predictions.]. Note that prediction is much faster than training, taking a few minutes with a decent GPU, and a few hours otherwise.

<h1>Estimating Feature Importance with SPEID</h1>

By predicting interactions between arbitrary enhancer and promoter sequences, SPEID allows us to measure the effects of any particular sequence feature on EPI prediction, providing a general method for evaluating the effects of <em>any</em> sequence modification. We specifically applied this to measure importance of human sequence motifs from the HOCOMOCOv10 database (http://hocomoco.autosome.ru/), by measuring the change in prediction performance when all occurences of that feature are removed (i.e., replaced with noise). Our implementation depended on the FIMO motif scanning algorithm, included as part of the MEME-Suite collection (http://meme-suite.org/tools/fimo). The pairwise/FIMO directory contains code required for estimating feature importances, given a set of feature locations. Note that this step can be computationally intensive, as it requries running the model prediction on all test data points (for each sequence feature, cell line, etc.). The feature_importances directory contains MATLAB code for then analyzing the results.
