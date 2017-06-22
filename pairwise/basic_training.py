# Basic python and data processing imports
import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
import h5py

import load_data_pairs as ld # my scripts for loading data
import build_small_model as bm # Keras specification of SPEID model

# import matplotlib.pyplot as plt
from datetime import datetime
import util

# Keras imports
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']

# Model training parameters
num_epochs = 32
batch_size = 100
training_frac = 0.9 # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
opt = Adam(lr = 1e-5) # opt = RMSprop(lr = 1e-6)

data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/original/all_data.h5'

for cell_line in cell_lines:
  print 'Loading ' + cell_line + ' data from ' + data_path
  X_enhancers = None
  X_promoters = None
  labels = None
  with h5py.File(data_path, 'r') as hf:
    X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
    X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
    labels = np.array(hf.get(cell_line + 'labels'))

  model = bm.build_model(use_JASPAR = False)

  model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics = ["accuracy"])

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
          # plt.ion()
  
  #     def on_batch_end(self, batch, logs = {}):
  #         self.training_losses.append(logs.get('loss'))
  #         self.training_accs.append(logs.get('acc'))
  
      def on_epoch_end(self, batch, logs = {}):
          self.training_losses.append(logs.get('loss'))
          self.training_accs.append(logs.get('acc'))
          self.epoch += 1
          val_predict = model.predict_classes([X_enhancers, X_promoters], batch_size = batch_size, verbose = 0)
          util.print_live(self, labels, val_predict, logs)
          # if self.epoch > 1: # need at least two time points to plot
          #     util.plot_live(self)
  
  # print '\nlabels.mean(): ' + str(labels.mean())
  print 'Data sizes: '
  print '[X_enhancers, X_promoters]: [' + str(np.shape(X_enhancers)) + ', ' + str(np.shape(X_promoters)) + ']'
  print 'labels: ' + str(np.shape(labels))

  # Instantiate callbacks
  confusionMatrix = ConfusionMatrix()
  checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/test-delete-this-" + cell_line + "-basic-" + t + ".hdf5"
  checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                  verbose=1)

  print 'Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...'
  model.fit([X_enhancers, X_promoters],
              [labels],
              # validation_data = ([X_enhancer, X_promoter], y_val),
              batch_size = batch_size,
              nb_epoch = num_epochs,
              shuffle = True,
              callbacks=[confusionMatrix, checkpointer]
              )
  
  # plotName = cell_line + '_' + t + '.png'
  # plt.savefig(plotName)
  # print 'Saved loss plot to ' + plotName
