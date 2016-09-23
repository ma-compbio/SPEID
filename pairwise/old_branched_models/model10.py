# Basic python and data processing imports
import numpy as np
import h5py
import scipy.io
from datetime import datetime
import load_data_pairs as ld # my own scripts for loading data
# np.random.seed(1337) # for reproducibility

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
# from keras.utils.visualize_util import plot
from seq2seq.layers.bidirectional import Bidirectional

# training parameters
num_epochs = 64
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel-" + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + ".hdf5"

# model parameters
n_kernels = 1024 # Number of kernels; used to be 1024
LSTM_out_dim = 32 # 128 # Output direction of ONE DIRECTION of LSTM; used to be 512

data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/K562/K562_ep_split.h5'
print 'Loading data from ' + data_path
X_enhancers_train, X_promoters_train, y_train = ld.load_hdf5_ep_split(data_path)

print 'Building model...'
# Convolutional/maxpooling layers to extract prominent motifs
# Separate but initially identical convolutional layers are trained for
# enhancers and promoters
# Enhancer branch:
enhancer_branch = Sequential()
enhancers_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_enhancers_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = 30,
                                        border_mode = "valid",
                                        subsample_length = 1)
enhancer_branch.add(enhancers_conv_layer)
enhancer_branch.add(MaxPooling1D(pool_length = 15, stride = 15))
enhancer_branch.add(Dropout(0.25))
# Bidirectional LSTM to extract combinations of motifs
enhancer_branch.add(Bidirectional(LSTM(output_dim=LSTM_out_dim,
                                    return_sequences=True)))
enhancer_branch.add(Dropout(0.5))

# Promoter branch:
promoter_branch = Sequential()
promoters_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_promoters_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = 30,
                                        border_mode = "valid",
                                        subsample_length = 1)
promoter_branch.add(promoters_conv_layer)
promoter_branch.add(MaxPooling1D(pool_length = 15, stride = 15))
promoter_branch.add(Dropout(0.25))
# Bidirectional LSTM to extract combinations of motifs
promoter_branch.add(Bidirectional(LSTM(output_dim=LSTM_out_dim,
                                    return_sequences=True)))
promoter_branch.add(Dropout(0.5))

# A single downstream model merges the enhancer and promoter branches
model = Sequential()

# Concatenate outputs of enhancer and promoter convolutional layers
model.add(Merge([enhancer_branch, promoter_branch],
                        mode='concat',
                        concat_axis = 1))

# Dense layer to allow nonlinearities
model.add(Flatten())
model.add(Dense(output_dim = 925, # used to be 925
                activation = "relu"))

# Logistic regression layer to make final binary prediction
model.add(Dense(output_dim = 1, activation = "sigmoid"))

# Read in and initialize convolutional layer with motifs from JASPAR
JASPAR_motifs = list(np.load('/home/sss1/Desktop/projects/DeepInteractions/JASPAR_CORE_2016_vertebrates.npy'))
print 'Initializing ' + str(len(JASPAR_motifs)) + ' kernels with JASPAR motifs.'
enhancers_conv_weights = enhancers_conv_layer.get_weights()
promoters_conv_weights = promoters_conv_layer.get_weights()
reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1],
JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1],
JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
JASPAR_motifs = JASPAR_motifs + reverse_motifs
for i in xrange(len(JASPAR_motifs)):
    m = JASPAR_motifs[i][::-1,:]
    w = len(m)
    start = np.random.randint(low=3, high=30-w+1-3)
    enhancers_conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
    enhancers_conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
    promoters_conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
    promoters_conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
enhancers_conv_layer.set_weights(enhancers_conv_weights)
promoters_conv_layer.set_weights(promoters_conv_weights)

print 'Compiling model...'
opt = RMSprop(lr = 1e-6)
model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics=["accuracy"])

# Print a summary of the model
model.summary()

print 'Running at most ' + str(num_epochs) + ' epochs...'

# Save a graph visualization of the model
# plot(model,
#         '/home/sss1/Desktop/projects/DeepInteractions/pairwise/model9.png',
#         show_shapes = True)

checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                verbose=1,
                                save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
model.fit([X_enhancers_train, X_promoters_train], 
            [y_train],
            batch_size=100, # used to be 100
            nb_epoch=num_epochs,
            shuffle=True,
            validation_split=0.09090909,
            callbacks=[checkpointer, earlystopper])
