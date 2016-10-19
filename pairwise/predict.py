import numpy as np
import h5py
import matplotlib.pyplot as plt

import build_model
import load_data_pairs as ld # my own scripts for loading data
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from keras.optimizers import Adam # not used but needed to compile model

cell_line = 'K562'
predictions_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line + '_predictions.h5'
already_computed = True

print 'Loading test data...'
# data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line + '_ep_split.h5'
data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line + '_ep_split.h5'
X_enhancer, X_promoter, y = ld.load_hdf5_ep_split(data_path)

if not already_computed:
  print 'Building model...'
  model = build_model.build_model(use_JASPAR = False)
  
  print 'Compiling model...'
  opt = Adam(lr = 1e-5) # opt = RMSprop(lr = 1e-6)
  model.compile(loss = 'binary_crossentropy',
                  optimizer = opt,
                  metrics = ["accuracy"])
  
  model.load_weights('/home/sss1/Desktop/projects/DeepInteractions/weights/balancedK562-noJASPAR-2016-10-13-18:40:51.hdf5')
  
  
  print 'Running predictions...'
  y_score = model.predict([X_enhancer, X_promoter], batch_size = 100, verbose = 1)
  
  
  print 'Saving predictions...'
  with h5py.File(predictions_path, 'w') as hf:
    hf.create_dataset('y_score', data = y_score)
  print 'Saved predictions to ' + predictions_path
else:
  print 'Loading predictions...'
  with h5py.File(predictions_path, 'r') as hf:
    y_score = np.array(hf.get('y_score'))

print 'Computing Precision-Recall curve...'
print 'Real shape: ' + str(np.shape(y)) + '     Prediction shape: ' + str(y_score)
precision, recall, _ = precision_recall_curve(y, y_score)
average_precision = average_precision_score(y, y_score)

print 'Precision:' + str(precision)
print 'Recall:' + str(recall)


print 'Plotting ROC curve...'
plt.figure()
lw = 2 # plot linewidth
plt.clf()
plt.plot(recall, precision, lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
plt.legend(loc="lower left")
plt.show()


# print 'Computing ROC curve...'
# print 'Real shape: ' + str(np.shape(y)) + '     Prediction shape: ' + str(y_score)
# fpr, tpr, thresholds = roc_curve(y, y_score)
# roc_auc = auc(fpr, tpr)
# 
# print 'Plotting ROC curve...'
# plt.figure()
# linewidth = 2
# plt.plot(fpr, tpr, color = 'darkorange',
#          lw = linewidth, label = 'ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color = 'navy', lw = linewidth, linestyle = '--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc = "lower right")
# plt.show()
