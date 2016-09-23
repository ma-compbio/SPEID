# Basic python and data processing imports
import numpy as np
import h5py
import scipy.io
from sklearn.utils import shuffle
from datetime import datetime
import load_data_pairs as ld # my own scripts for loading data
# np.random.seed(1337) # for reproducibility

# Keras imports
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from seq2seq.layers.bidirectional import Bidirectional

# training parameters
num_epochs = 64
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel-" + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + ".hdf5"

# model parameters
n_kernels = 1024 # Number of kernels; used to be 1024
LSTM_out_dim = 128 # Output direction of ONE DIRECTION of LSTM; used to be 512

data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/K562/K562.h5'
print 'Loading data from ' + data_path
X_train, y_train = ld.load_hdf5(data_path)

# manually shuffle the data (since keras does not shuffle validation data)
print 'Shuffling data...'
rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

print 'Building model...'
model = Sequential()
conv_layer = Convolution1D(input_dim=4,
                        input_length=X_train.shape[1],
                        nb_filter=n_kernels,
                        filter_length=30,
                        border_mode="valid",
                        subsample_length=1)
brnn = Bidirectional(LSTM(input_dim=n_kernels,
                        output_dim=LSTM_out_dim,
                        return_sequences=True))

# Construct network
model.add(conv_layer)
model.add(MaxPooling1D(pool_length = 15, stride = 15))
model.add(Dropout(0.2))
model.add(brnn)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(output_dim = 500, # used to be 925
                    activation = "relu"))
model.add(Dense(output_dim = 1,
                    activation = "sigmoid"))

# Read in and initialize convolutional layer with motifs from JASPAR
JASPAR_motifs = list(np.load('/home/sss1/Desktop/projects/DeepInteractions/JASPAR_CORE_2016_vertebrates.npy'))
print 'Initializing ' + str(len(JASPAR_motifs)) + ' kernels with JASPAR motifs.'
conv_weights = conv_layer.get_weights()
reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1],
JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1],
JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
JASPAR_motifs = JASPAR_motifs + reverse_motifs
for i in xrange(len(JASPAR_motifs)):
    m = JASPAR_motifs[i][::-1,:]
    w = len(m)
    start = np.random.randint(low=3, high=30-w+1-3)
    conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
    conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
conv_layer.set_weights(conv_weights)

print 'Compiling model...'
opt = RMSprop(lr = 1e-6)
model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                # class_mode = "binary",
                metrics=["accuracy"])

print 'Running at most ' + str(num_epochs) + ' epochs...'

checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                verbose=1,
                                save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
model.fit(X_train,
            y_train,
            batch_size=100, # used to be 100
            nb_epoch=num_epochs,
            shuffle=True,
            validation_split=0.09090909,
            callbacks=[checkpointer, earlystopper])
