import numpy as np
import h5py
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
# import matplotlib.pyplot as plt

from keras.optimizers import Adam # not used but needed to compile model

test_cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
for test_cell_line in test_cell_lines:

  data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/' + test_cell_line + '_ep_split.h5'
  predictions_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + test_cell_line + '/train_small' + test_cell_line + '_test_' + test_cell_line + '_predictions.h5'
  out_path = '/home/sss1/Desktop/projects/DeepInteractions/pairwise/tmp_data/' + test_cell_line + '.csv'

  # print 'Loading labels from ' + data_path + '...'
  with h5py.File(data_path, 'r') as hf:
    y = np.array(hf.get('y_train'))
  # print 'Loading predictions from ' + predictions_path + '...'
  with h5py.File(predictions_path, 'r') as hf:
    y_score = np.squeeze(np.array(hf.get('y_score')))

  # print 'y: ' + str(y)
  # print 'y.shape(): ' + str(np.shape(y))
  # # print 'mean(y) :' + str(np.mean(y))
  # print 'y_score: ' + str(y_score)
  # print 'y_score.shape(): ' + str(np.shape(y_score))
  # # print 'mean(y_score) :' + str(np.mean(y_score))
  # print 'corrcoef(y, y_score) :' + str(np.corrcoef(y, y_score))

  # precision, recall, thresholds = precision_recall_curve(y, y_score)
  print '\nCell line: ' + test_cell_line
  fpr, tpr, _ = roc_curve(y, y_score)
  print 'AUPR: ' + str(average_precision_score(y, y_score))
  print 'AUROC: ' + str(auc(fpr, tpr))
  # 
  # plt.clf()
  # # plt.plot(fpr, tpr)
  # plt.plot(recall, precision)
  # plt.xlabel('Recall')
  # plt.ylabel('Precision')
  # plt.ylim([0.0, 1.05])
  # plt.xlim([0.0, 1.0])
  # plt.show()

  output = np.transpose(np.vstack((y, y_score)))
  print 'Combined shape:' + str(np.shape(output))

  np.savetxt(out_path, output, delimiter = ",")
