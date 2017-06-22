import util
# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
# from seq2seq.layers.bidirectional import Bidirectional
from keras.regularizers import l2, activity_l2

# model parameters
enhancer_length = 3000 # TODO: get this from input
promoter_length = 2000 # TODO: get this from input
n_kernels = 1024 # Number of kernels; used to be 1024
filter_length = 40 # Length of each kernel
LSTM_out_dim = 100 # Output direction of ONE DIRECTION of LSTM; used to be 512


# Convolutional/maxpooling layers to extract prominent motifs
# Separate identically initialized convolutional layers are trained for
# enhancers and promoters
# Define enhancer layers
enhancer_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = enhancer_length,
                                        nb_filter = n_kernels,
                                        filter_length = filter_length,
                                        border_mode = "valid",
                                        subsample_length = 1,
                                        W_regularizer = l2(1e-6))
enhancer_max_pool_layer = MaxPooling1D(pool_length = filter_length/2, stride = filter_length/2)

# Build enhancer branch
enhancer_branch = Sequential()
enhancer_branch.add(enhancer_conv_layer)
# enhancer_branch.add(Activation("relu"))
enhancer_branch.add(enhancer_max_pool_layer)

# Define promoter layers branch:
promoter_conv_layer = Convolution1D(input_dim = 4,
                                        input_length = promoter_length,
                                        nb_filter = n_kernels,
                                        filter_length = filter_length,
                                        border_mode = "valid",
                                        subsample_length = 1,
                                        W_regularizer = l2(1e-6))
promoter_max_pool_layer = MaxPooling1D(pool_length = filter_length/2, stride = filter_length/2)

# Build promoter branch
promoter_branch = Sequential()
promoter_branch.add(promoter_conv_layer)
# promoter_branch.add(Activation("relu"))
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
dense_layer = Dense(output_dim = 1000,
                    init = "glorot_uniform",
                    W_regularizer = l2(1e-6))

# Logistic regression layer to make final binary prediction
LR_classifier_layer = Dense(output_dim = 1)
  

def build_model(use_JASPAR = True):

  # A single downstream model merges the enhancer and promoter branches
  # Build main (merged) branch
  # Using batch normalization seems to inhibit retraining, probably because the
  # point of retraining is to learn (external) covariate shift
  model = Sequential()
  model.add(merge_layer)
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(biLSTM_layer)
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(dense_layer)
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(LR_classifier_layer)
  model.add(BatchNormalization())
  model.add(Activation("sigmoid"))
  
  # Read in and initialize convolutional layers with motifs from JASPAR
  if use_JASPAR:
    util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)

  return model

def build_frozen_model():

  # Freeze all by the dense layers of the network
  enhancer_conv_layer.trainable = False
  enhancer_max_pool_layer.trainable = False
  promoter_conv_layer.trainable = False
  promoter_max_pool_layer.trainable = False
  biLSTM_layer.trainable = False

  # TODO: Figure out how to remove layers after loading weights

  model = Sequential()
  model.add(merge_layer)
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(biLSTM_layer)
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(dense_layer)
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(LR_classifier_layer)
  model.add(BatchNormalization())
  model.add(Activation("sigmoid"))

  return model
