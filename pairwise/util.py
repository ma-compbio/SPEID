import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss, roc_curve, auc, precision_recall_curve, average_precision_score

def initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer):
    JASPAR_motifs = list(np.load('/home/sss1/Desktop/projects/DeepInteractions/JASPAR_CORE_2016_vertebrates.npy'))
    print 'Initializing ' + str(len(JASPAR_motifs)) + ' kernels with JASPAR motifs.'
    enhancer_conv_weights = enhancer_conv_layer.get_weights()
    promoter_conv_weights = promoter_conv_layer.get_weights()
    reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1],
    JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1],
    JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
    JASPAR_motifs = JASPAR_motifs + reverse_motifs
    for i in xrange(len(JASPAR_motifs)):
        m = JASPAR_motifs[i][::-1,:]
        w = len(m)
        start = np.random.randint(low=3, high=30-w+1-3)
        enhancer_conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
        enhancer_conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
        promoter_conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
        promoter_conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)
    enhancer_conv_layer.set_weights(enhancer_conv_weights)
    promoter_conv_layer.set_weights(promoter_conv_weights)

# Splits the data into training and validation data, keeping training_frac of
# the input samples in the training set and the rest for validation
def split_train_and_val_data(X_enhancer_train, X_promoter_train, y_train, training_frac):

    n_train = int(training_frac * np.shape(y_train)[0]) # number of training samples

    X_enhancer_val = X_enhancer_train[n_train:, :]
    X_enhancer_train = X_enhancer_train[:n_train, :]

    X_promoter_val = X_promoter_train[n_train:, :]
    X_promoter_train = X_promoter_train[:n_train, :]

    y_val = y_train[n_train:]
    y_train = y_train[:n_train]

    return X_enhancer_train, X_promoter_train, y_train, X_enhancer_val, X_promoter_val, y_val

# Calculates and prints several metrics (confusion matrix, Precision/Recall/F1)
# in real time; also updates the values in the conf_mat_callback so they can be
# plotted or analyzed later
def print_live(conf_mat_callback, y_val, val_predict, logs):

    conf_mat = confusion_matrix(y_val, val_predict).astype(float)

    precision = conf_mat[1, 1] / conf_mat[:, 1].sum()
    recall = conf_mat[1, 1] / conf_mat[1, :].sum()
    f1_score = 2 * precision * recall / (precision + recall)

    acc = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)

    loss = log_loss(y_val, val_predict)

    conf_mat_callback.precisions.append(precision)
    conf_mat_callback.recalls.append(recall)
    conf_mat_callback.f1_scores.append(f1_score)
    conf_mat_callback.losses.append(loss)
    conf_mat_callback.accs.append(acc)
    print '\nConfusion matrix:\n' + str(conf_mat) + '\n'
    print 'Precision: ' + str(precision) + \
        '  Recall: ' + str(recall) + \
        '  F1: ' + str(f1_score) + \
        '  Accuracy: ' + str(acc) + \
        '  Log Loss: ' + str(loss)
    print 'Predicted fractions: ' + str(val_predict.mean())
    print 'Actual fractions: ' + str(y_val.mean()) + '\n'

# Plots several metrics (Precision/Recall/F1, loss, Accuracy) in real time
def plot_live(conf_mat_callback):

    epoch = conf_mat_callback.epoch

    plt.clf()
    xs = [1 + i for i in range(epoch)]
    precisions_plot = plt.plot(xs, conf_mat_callback.precisions, label = 'Precision')
    recalls_plot = plt.plot(xs, conf_mat_callback.recalls, label = 'Recall')
    f1_scores_plot = plt.plot(xs, conf_mat_callback.f1_scores, label = 'F1 score')
    accs_plot = plt.plot(xs, conf_mat_callback.accs, label = 'Accuracy')
    losses_plot = plt.plot(xs, conf_mat_callback.losses / max(conf_mat_callback.losses), label = 'Loss')
    batch_xs = [1 + epoch * float(i)/len(conf_mat_callback.training_losses) for i in range(len(conf_mat_callback.training_losses))]
    training_losses_plot = plt.plot(batch_xs, conf_mat_callback.training_losses / max(conf_mat_callback.training_losses), label = 'Training Loss')
    training_losses_plot = plt.plot(batch_xs, conf_mat_callback.training_accs, label = 'Training Accuracy')
    plt.legend(bbox_to_anchor = (0, 1), loc = 4, borderaxespad = 0., prop={'size':6})
    plt.ylim([0, 1])
    plt.pause(.001)

# Given a (nearly) balanced data set (i.e., labeled enhancer and promoter
# sequence pairs), subsamples the positive samples to produce the desired
# fraction of positive samples; retains all negative samples
def subsample_imbalanced(X_enhancer, X_promoter, y, positive_subsample_frac):
    n = np.shape(y_train)[0] # sample size (i.e., number of pairs)

    # indices that are positive and selected to be retained or negative
    to_keep = (np.random(n) < positive_subsample_frac) or (y == 1)

    return X_enhancer[to_keep, :], X_promoter[to_keep, :], y[to_keep]


def compute_AUPR(y, y_score):
  # print 'Computing Precision-Recall curve...'
  precision, recall, _ = precision_recall_curve(y, y_score)
  average_precision = average_precision_score(y, y_score)

def plot_PR_curve(y, y_score):
  # print 'Computing Precision-Recall curve...'
  precision, recall, _ = precision_recall_curve(y, y_score)
  return average_precision_score(y, y_score)

def plot_ROC_curve(y, y_score):
  # print 'Computing ROC curve...'
  fpr, tpr, thresholds = roc_curve(y, y_score)
  return auc(fpr, tpr)
