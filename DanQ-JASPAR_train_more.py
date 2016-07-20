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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from seya.layers.recurrent import Bidirectional

# PARAMETERS
num_epochs = 9
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel.hdf5"

print 'building model'

# First convolutional layer
conv_layer = Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=1024,
                        filter_length=30,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1)


# Define bidirectional LSTM layer
forward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
backward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True, go_backwards=True)
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

# Construct network
model = Sequential()
model.add(conv_layer)
model.add(MaxPooling1D(pool_length=15, stride=15))
model.add(Dropout(0.2))
model.add(brnn)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(input_dim=64*1024, output_dim=925, activation="relu"))
model.add(Dense(input_dim=925, output_dim=919, activation="sigmoid"))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
class_mode="binary")

weights_file = '/home/sss1/Desktop/projects/DeepInteractions/weights/myDanQ-JASPAR_bestmodel.hdf5'
print 'loading weights from ' + weights_file + '...'
model.load_weights(weights_file)

print 'loading data'
data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/deepsea/deepsea_train/'
trainmat = h5py.File(data_path + 'train.mat')
validmat = scipy.io.loadmat(data_path + 'valid.mat')
testmat = scipy.io.loadmat(data_path + 'test.mat')
X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T

print 'running at most ' + str(num_epochs) + ' epochs'

checkpointer = ModelCheckpoint(filepath=checkpoint_path,
verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

model.fit(X_train, y_train, batch_size=100, nb_epoch=num_epochs, shuffle=True,
show_accuracy=True,
validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)),
validmat['validdata']), callbacks=[checkpointer,earlystopper])

tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)),
testmat['testdata'],show_accuracy=True)

print tresults
