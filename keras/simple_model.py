import numpy as np
import h5py
import scipy.io
np.random.seed(0) # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

print 'loading data'

# For now, use mini training set (1/100 size)
# trainmat = h5py.File('data/train.mat')
# X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
# y_train = np.array(trainmat['traindata']).T

trainmat = scipy.io.loadmat('data/train_small.mat')
X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(0, 2, 1))
y_train = np.array(trainmat['traindata'])
validmat = scipy.io.loadmat('data/valid.mat')
testmat = scipy.io.loadmat('data/test.mat')

print 'building model'

model = Sequential()

# Add one convolutional/max-pooling layer, with 20% dropout
model.add(Convolution1D(input_dim=4, # one-hot coding of ATCG
                        input_length=1000, # data comes in 1kbp windows
                        nb_filter=320, # number of kernels
                        filter_length=26, # kernel width
                        border_mode="valid", # something about padding input
                        activation="relu", # rectified linear unit; dunno why
                        subsample_length=1)) # don't subsample
model.add(MaxPooling1D(pool_length=13, stride=13))
model.add(Dropout(0.2))

# Add two dense layers
model.add(Flatten())
model.add(Dense(input_dim=75*640, output_dim=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))

print 'compiling model...'
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print 'running at most 60 epochs...'

checkpointer = ModelCheckpoint(filepath="best_model.hdf5",
                               verbose=1,
                               save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

t_start_train = time.time()

model.fit(X_train,
          y_train,
          batch_size=100,
          nb_epoch=30, # TODO: change to 60
          verbose=1, # show progress bar
          shuffle=True,
          show_accuracy=True,
          validation_data=(np.transpose(validmat['validxdata'], axes=(0,2,1)),
                           validmat['validdata']),
          callbacks=[checkpointer, earlystopper])

training_time = time.time() - t_start_train()
t_start_test = time.time()

test_results = model.evaluate(np.transpose(testmat['testxdata'],
                                           axes=(0,2,1)),
                              testmat['testdata'],
                              show_accuracy=True)

testing_time = time.time() - t_start_test()

print test_results
print "Total time taken: " + str(training_time + testing_time) + " seconds"
print "Training time taken: " + str(training_time) + " seconds"
print "Testing time taken: " + str(testing_time) + " seconds"
