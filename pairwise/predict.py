import numpy as np
import h5py
import matplotlib.pyplot as plt
import util

import build_small_model as bm
import load_data_pairs as ld # my own scripts for loading data

from keras.optimizers import Adam # not used but needed to compile model

def predict(train_cell_line, test_cell_line):

  data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/' + test_cell_line + '_ep_split.h5'
  predictions_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/train_small' + train_cell_line + '_test_' + test_cell_line + '_predictions.h5'

  print 'Loading test data...'
  X_enhancer, X_promoter, y = ld.load_hdf5_ep_split(data_path)
  
  print 'Building model...'
  model = bm.build_model(use_JASPAR = False)
  
  print 'Compiling model...'
  opt = Adam(lr = 1e-5)
  model.compile(loss = 'binary_crossentropy',
                  optimizer = opt,
                  metrics = ["accuracy"])
  
  print 'Loading ' + train_cell_line + ' model weights...'
  model.load_weights('/home/sss1/Desktop/projects/DeepInteractions/weights/best/small_model_balanced' + train_cell_line + '-noJASPAR.hdf5')
  
  print 'Running predictions...'
  y_score = model.predict([X_enhancer, X_promoter], batch_size = 50, verbose = 1)
  
  print 'Saving predictions...'
  with h5py.File(predictions_path, 'w') as hf:
    hf.create_dataset('y_score', data = y_score)
  print 'Saved predictions to ' + predictions_path


def plot_PR_and_ROC_curves(train_cell_line, test_cell_line):
  data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/' + test_cell_line + '_ep_split.h5'
  predictions_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/train_small' + train_cell_line + '_test_' + test_cell_line + '_predictions.h5'
  print 'True label path: ' + data_path
  print 'Prediction score path: ' + predictions_path

  # # print 'Loading labels from ' + data_path + '...'
  # with h5py.File(data_path, 'r') as hf:
  #   y = np.array(hf.get('y_train'))
  # # print 'Loading predictions from ' + predictions_path + '...'
  # with h5py.File(predictions_path, 'r') as hf:
  #   y_score = np.array(hf.get('y_score'))
  
  # # ap = util.plot_PR_curve(y, y_score)
  # # print '(' + train_cell_line + ', ' + test_cell_line + ') AUPR: ' + str(round(ap, 2))
  # roc_auc = util.plot_ROC_curve(y, y_score)
  # print '(' + train_cell_line + ', ' + test_cell_line + ') AUROC: ' + str(round(roc_auc, 2))

train_cell_lines = ['4lines']# ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
test_cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
for train_cell_line in train_cell_lines:
  for test_cell_line in test_cell_lines:
    plot_PR_and_ROC_curves(train_cell_line, test_cell_line)
    # print '\nPredicting ' + test_cell_line + ' after training on ' + train_cell_line 
    # predict(train_cell_line, test_cell_line)
  print ''
