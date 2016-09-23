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
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
# from keras.utils.visualize_util import plot
from seq2seq.layers.bidirectional import Bidirectional

# training parameters
num_epochs = 5
num_epochs_frozen = 16
batch_size = 100
training_frac = 0.9 # use 90% of data for training, 10% for testing
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel-" + t + ".hdf5"

# model parameters
n_kernels = 1024 # Number of kernels; used to be 1024
filter_length = 40 # Length of each kernel
LSTM_out_dim = 100 # Output direction of ONE DIRECTION of LSTM; used to be 512

# Load data and split into training and validation sets
cell_line = 'K562'
data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line + '_ep_split.h5'
print 'Loading data from ' + data_path
X_enhancer_train, X_promoter_train, y_train = ld.load_hdf5_ep_split_aug(data_path)
X_enhancer_train, X_promoter_train, y_train, X_enhancer_val, X_promoter_val, y_val = util.split_train_and_val_data(X_enhancer_train, X_promoter_train, y_train, training_frac)

print 'Building model...'
# Convolutional/maxpooling layers to extract prominent motifs
# Separate identically initialized convolutional layers are trained for
# enhancers and promoters
# Define enhancer layers
enhancer_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_enhancer_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = filter_length,
                                        border_mode = "valid",
                                        subsample_length = 1)
enhancer_max_pool_layer = MaxPooling1D(pool_length = filter_length/2, stride = filter_length/2)

# Build enhancer branch
enhancer_branch = Sequential()
enhancer_branch.add(enhancer_conv_layer)
enhancer_branch.add(enhancer_max_pool_layer)

# Define promoter layers branch:
promoter_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = X_promoter_train.shape[1],
                                        nb_filter = n_kernels,
                                        filter_length = filter_length,
                                        border_mode = "valid",
                                        subsample_length = 1)
promoter_max_pool_layer = MaxPooling1D(pool_length = filter_length/2, stride = filter_length/2)

# Build promoter branch
promoter_branch = Sequential()
promoter_branch.add(promoter_conv_layer)
promoter_branch.add(promoter_max_pool_layer)

# Define main model layers
# Concatenate outputs of enhancer and promoter convolutional layers
merge_layer = Merge([enhancer_branch, promoter_branch],
                        mode = 'concat',
                        concat_axis = 1)


# Bidirectional LSTM to extract combinations of motifs
biLSTM_layer = Bidirectional(LSTM(input_dim = n_kernels,
                                    output_dim = LSTM_out_dim,
                                    return_sequences = True))

# Dense layer to allow nonlinearities
dense_layer = Dense(output_dim = 1000, init = "glorot_uniform")

# Logistic regression layer to make final binary prediction
LR_classifier_layer = Dense(output_dim = 1)

### BEGIN CODE FOR TRAINING MODEL ON BALANCED DATA ###
# A single downstream model merges the enhancer and promoter branches
# Build main (merged) branch
model = Sequential()
model.add(merge_layer)
model.add(Dropout(0.25))
model.add(biLSTM_layer)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(dense_layer)
# model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(LR_classifier_layer)
# model.add(BatchNormalization())
model.add(Activation("sigmoid"))

# Read in and initialize convolutional layers with motifs from JASPAR
util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)

print 'Compiling model...'
# opt = RMSprop(lr = 1e-6)
opt = Adam(lr = 1e-5)
model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics = ["accuracy"])

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
print 'Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...'
model.fit([X_enhancer_train, X_promoter_train],
            [y_train],
            batch_size = batch_size,
            nb_epoch = num_epochs,
            shuffle = True,
            callbacks=[confusionMatrix]
            )

plotName = 'EP_' + t + '.png'
plt.savefig(plotName)
print 'Saved loss plot to ' + plotName

### BEGIN CODE FOR RETRAINING MODEL ON IMBALANCED DATA ###
# Freeze all by the dense layers of the network
enhancer_conv_layer.trainable = False
enhancer_max_pool_layer.trainable = False
promoter_conv_layer.trainable = False
promoter_max_pool_layer.trainable = False
biLSTM_layer.trainable = False


print 'Compiling retraining model...'
model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics=["accuracy"])

# subsample balanced training data to create imbalanced training data
X_enhancer_train, X_promoter_train, y_train = ld.load_hdf5_ep_split(data_path)
X_enhancer_train, X_promoter_train, y_train, X_enhancer_val, X_promoter_val, y_val = util.split_train_and_val_data(X_enhancer_train, X_promoter_train, y_train, training_frac)

# fraction of samples in each class
pos_frac = y_train.mean()
neg_weight = 1/(1 - pos_frac)
pos_weight = 1/pos_frac

# Instantiate callbacks for frozen training
confusionMatrixFrozen = ConfusionMatrix()
print 'Running partly frozen model for exactly ' + str(num_epochs_frozen) + ' epochs...'
model.fit([X_enhancer_train, X_promoter_train],
            [y_train],
            batch_size = batch_size,
            nb_epoch = num_epochs_frozen,
            shuffle = True,
            callbacks=[confusionMatrixFrozen],
            class_weight = {0 : neg_weight, 1 : pos_weight} # increase weight of positive samples, to counter class imbalance
            )

plotName = 'EP_frozen_' + t + '.png'
plt.savefig(plotName)
print 'Saved loss plot to ' + plotName
