# Basic python and data processing imports
import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py
import scipy.io
from datetime import datetime
import load_data_pairs as ld # my own scripts for loading data
import build_model

# Save plot of errors, but don't display it live
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util
# np.random.seed(1337) # for reproducibility

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback# , ReduceLROnPlateau
from keras.models import Sequential
from seq2seq.layers.bidirectional import Bidirectional

# training parameters
cell_line = 'HUVEC'
num_epochs = 40
batch_size = 64
training_frac = 0.9 # use 90% of data for training, 10% for testing/validation
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
checkpoint_path =
"/home/sss1/Desktop/projects/DeepInteractions/weights/balanced" + cell_line + "-noJASPAR-" + t + ".hdf5"
opt = Adam(lr = 1e-5) # opt = RMSprop(lr = 1e-6)

# # Load data and split into training and validation sets
data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line + '_ep_split.h5'
print 'Loading data from ' + data_path

X_enhancer_train, X_promoter_train, y_train = ld.load_hdf5_ep_split_aug(data_path)
X_enhancer_train, X_promoter_train, y_train, X_enhancer_val, X_promoter_val, y_val = util.split_train_and_val_data(X_enhancer_train, X_promoter_train, y_train, training_frac)

print 'Building model...'
model = build_model.build_model(use_JASPAR = False)

# model.load_weights('/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel-2016-10-08-01:02:54.hdf5')

print 'Compiling model...'
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
        self.training_losses = []
        self.training_accs = []
        self.accs = []
        plt.ion()

    def on_epoch_end(self, batch, logs = {}):
        self.training_losses.append(logs.get('loss'))
        self.training_accs.append(logs.get('acc'))
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
checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                verbose=1,
                                save_best_only=True)

print 'Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...'
model.fit([X_enhancer_train, X_promoter_train],
            [y_train],
            validation_data = ([X_enhancer_val, X_promoter_val], y_val),
            batch_size = batch_size,
            nb_epoch = num_epochs,
            shuffle = True,
            callbacks=[confusionMatrix, checkpointer]
            )

plotName = cell_line + '_' + t + '.png'
plt.savefig(plotName)
print 'Saved loss plot to ' + plotName
