import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from seya.layers.recurrent import Bidirectional

import theano

import load_data

forward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
backward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm,
        return_sequences=True)

print 'building model...'

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=1024,
                        filter_length=30,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=15, stride=15))
model.add(Dropout(0.2))
model.add(brnn)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(input_dim=64*1024, output_dim=925, activation="relu"))
model.add(Dense(input_dim=925, output_dim=919, activation="sigmoid"))

print 'compiling model...'
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
        class_mode="binary")

weights_file = '/home/sss1/Desktop/projects/DeepInteractions/DanQ-JASPAR_bestmodel.hdf5'
print 'loading weights from ' + weights_file + '...'
model.load_weights(weights_file)

print 'loading input data...'
path = '/home/sss1/Desktop/projects/DeepInteractions'
file_root = 'pairs_epK562positive_seq2_sepwin'
# file_root = sys.argv[1] # USE COMMAND LINE ARG
suffix_in = '.txt'
data_path = path + file_root + suffix_in
Xs = np.transpose(load_data.load_and_format_data(data_path), axes=(0,2,1))
print 'Successfully loaded ' + str(Xs.shape[0]) + ' samples.'

print 'predicting on test sequences...'
Ys = model.predict(Xs, verbose=1)
print 'The output data has shape: ' + str(np.shape(Ys))
print 'The output data has mean: ' + str(np.mean(Ys.flatten()))

suffix_out = '.mat'
save_path = path + 'predictions_' + file_root + suffix_out
print "saving to " + save_path
scipy.io.savemat(save_path, mdict={'predicted' : Ys,
                                   'model' : 'DanQ_JASPAR',
                                   'data' : data_path})
