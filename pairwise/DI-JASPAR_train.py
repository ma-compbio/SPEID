# Basic pythong and data processing imports
import numpy as np
import h5py
import scipy.io
from sklearn.utils import shuffle
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
from seya.layers.recurrent import Bidirectional

# training parameters
num_epochs = 9
checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/myDanQ-JASPAR_bestmodel.hdf5"
# data parameters
enhancer_length = 1000
promoter_length = 1000

print 'Building model...'
model = Sequential()
conv_layer = Convolution1D(input_dim=4,
                        input_length=enhancer_length + promoter_length,
                        nb_filter=1024,
                        filter_length=30,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1)

conv_weights = conv_layer.get_weights()

# Read in and initialize convolutional layer with motifs from JASPAR
JASPAR_motifs = list(np.load('/home/sss1/Desktop/projects/DeepInteractions/JASPAR_CORE_2016_vertebrates.npy'))
reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1],
JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1],
JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
JASPAR_motifs = JASPAR_motifs + reverse_motifs

for i in xrange(len(JASPAR_motifs)):
    m = JASPAR_motifs[i][::-1,:]
    w = len(m)
    #conv_weights[0][i,:,:,0] = 0
    #start = (30-w)/2
    start = np.random.randint(low=3, high=30-w+1-3)
    conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
    #conv_weights[1][i] = -0.5
    conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
conv_layer.set_weights(conv_weights)

# Define bidirectional LSTM layer
forward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
backward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True, go_backwards=True)
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

# Construct network
model.add(conv_layer)
model.add(MaxPooling1D(pool_length=15, stride=15))
model.add(Dropout(0.2))
model.add(brnn)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(input_dim=64*1024, output_dim=925, activation="relu"))
model.add(Dense(input_dim=925, output_dim=1, activation="sigmoid"))

print 'Compiling model...'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

print 'Loading data sets...'
data_dir = '/home/sss1/Desktop/projects/DeepInteractions/data/pairs_ep/'
positive_enhancers = data_dir + 'K562_pos_1.txt'
positive_promoters = data_dir + 'K562_pos_2.txt'
negative_enhancers = data_dir + 'K562_neg_1.txt'
negative_promoters = data_dir + 'K562_neg_2.txt'
X_train, y_train = ld.load_full_data(positive_enhancers,
    positive_promoters,
    negative_enhancers,
    negative_promoters)
X_train = transpose(samples, axes=(0,2,1))

# manually shuffle the data (since keras does not shuffle validation data)
rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

print 'Running at most ' + str(num_epochs) + ' epochs...'

checkpointer = ModelCheckpoint(filepath=checkpoint_path,
verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

model.fit(X_train,
            y_train,
            batch_size=100,
            nb_epoch=num_epochs,
            shuffle=True,
            show_accuracy=True,
            validation_split=0.1,
            callbacks=[checkpointer, earlystopper])

# No test set, for now
# testmat = scipy.io.loadmat('deepsea_train/test.mat')
# tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)),
# testmat['testdata'],show_accuracy=True)
# print tresults
