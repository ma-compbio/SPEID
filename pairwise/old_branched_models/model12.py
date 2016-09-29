# Basic python and data processing imports
import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime
import load_data_pairs as ld # my own scripts for loading data
import matplotlib.pyplot as plt
import util
# np.random.seed(1337) # for reproducibility

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
# from keras.utils.visualize_util import plot
from seq2seq.layers.bidirectional import Bidirectional


# training parameters
num_epochs = 64
batch_size = 100
training_frac = 0.90909091
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel-" + t + ".hdf5"

# model parameters
n_kernels = 1024 # Number of kernels; used to be 1024
LSTM_out_dim = 64 # Output direction of ONE DIRECTION of LSTM; used to be 512

data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/K562/K562_ep_split.h5'
print 'Loading data from ' + data_path
X_enhancer_train, X_promoter_train, y_train = ld.load_hdf5_ep_split(data_path)

X_enhancer_train, X_promoter_train, y_train, X_enhancer_val, X_promoter_val, y_val = util.split_train_and_val_data(X_enhancer_train, X_promoter_train, y_train, training_frac)

print 'Building model...'
# Convolutional/maxpooling layers to extract prominent motifs
# Separate identically initialized convolutional layers are trained for
# enhancers and promoters
# Enhancer branch:
enhancer_branch = Sequential()
enhancer_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_enhancer_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = 60,
                                        border_mode = "valid",
                                        subsample_length = 1)
enhancer_branch.add(enhancer_conv_layer)
enhancer_branch.add(MaxPooling1D(pool_length = 30, stride = 30))

# Promoter branch:
promoter_branch = Sequential()
promoter_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_promoter_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = 60,
                                        border_mode = "valid",
                                        subsample_length = 1)
promoter_branch.add(promoter_conv_layer)
promoter_branch.add(MaxPooling1D(pool_length = 30, stride = 30))

# A single downstream model merges the enhancer and promoter branches
model = Sequential()

# Concatenate outputs of enhancer and promoter convolutional layers
model.add(Merge([enhancer_branch, promoter_branch],
                        mode='concat',
                        concat_axis = 1))
model.add(Dropout(0.25))

# Bidirectional LSTM to extract combinations of motifs
model.add(Bidirectional(LSTM(input_dim = n_kernels,
                                    output_dim = LSTM_out_dim,
                                    return_sequences = True)))
model.add(Dropout(0.5))

# Dense layer to allow nonlinearities
model.add(Flatten())
model.add(Dense(output_dim = 925, # used to be 925
                activation = "relu",
                init = "glorot_uniform"))

# Logistic regression layer to make final binary prediction
model.add(Dense(output_dim = 1, activation = "sigmoid"))

# Read in and initialize convolutional layers with motifs from JASPAR
util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)

print 'Compiling model...'
# opt = RMSprop(lr = 1e-6)
opt = Adam(lr = 1e-5)
model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics=["accuracy"])

# Print a summary of the model
model.summary()

# Define custom callback that prints/plots performance at end of each epoch
class ConfusionMatrix(Callback):
    def on_train_begin(self, logs = {}):
        self.epoch = 0
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.losses = []
        self.accs = []
        plt.ion()

    def on_epoch_end(self, batch, logs = {}):
        self.epoch += 1
        val_predict = model.predict_classes([X_enhancer_val, X_promoter_val], batch_size = batch_size, verbose = 0)
        util.print_live(self, y_val, val_predict, logs)
        if self.epoch > 1: # need at least two time points to plot
            util.plot_live(self)

print '\nData sizes: '
print '[X_enhancers_train, X_promoters_train]: [' + str(np.shape(X_enhancer_train)) + ', ' + str(np.shape(X_promoter_train)) + ']'
print 'y_train: ' + str(np.shape(y_train))
print 'y_train.mean(): ' + str(y_train.mean())

# Instantiate callbacks
confusionMatrix = ConfusionMatrix()
# checkpointer = ModelCheckpoint(filepath=checkpoint_path,
#                                 verbose=1,
#                                 save_best_only=True)
# earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
print 'Running at most ' + str(num_epochs) + ' epochs...'
model.fit([X_enhancer_train, X_promoter_train], 
            [y_train],
            batch_size = batch_size, # used to be 100
            nb_epoch = num_epochs,
            shuffle = True,
            callbacks=[confusionMatrix]
            )

plotName = 'EP_' + t + '.png'
plt.savefig(plotName)
print 'Saved loss plot to ' + plotName
