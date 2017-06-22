import numpy as np
import sys
import h5py

def load_hdf5(path):
  with h5py.File(path,'r') as hf:
    X_train = np.array(hf.get('X_train'))
    y_train = np.array(hf.get('y_train'))
    return X_train, y_train

def load_hdf5_ep_split(path):
  with h5py.File(path,'r') as hf:
    X_enhancers_train = np.array(hf.get('X_enhancers_train'))
    X_promoters_train = np.array(hf.get('X_promoters_train'))
    y_train = np.array(hf.get('y_train'))
    return X_enhancers_train, X_promoters_train, y_train

def load_hdf5_ep_split_aug(path):
  with h5py.File(path,'r') as hf:
    X_enhancers_train = np.array(hf.get('X_enhancers_train_aug'))
    X_promoters_train = np.array(hf.get('X_promoters_train_aug'))
    y_train = np.array(hf.get('y_train_aug'))
    return X_enhancers_train, X_promoters_train, y_train

# Reads the specified files and builds num_samples X 4 X seg_length arrays for
# both positive and negative versions of the enhancers and promoter data sets
# Outputs:
#   1) concatenations of enhancer and promoter segments, for all positive and
#   negative samples
#   2) labels (0 for negative, 1 for positive) for corresponding samples
def load_full_data(pos_enhancer_f_name, pos_promoter_f_name, neg_enhancer_f_name, neg_promoter_f_name):

  print 'Loading positive samples...'
  positive = load_ep_pairs(pos_enhancer_f_name, pos_promoter_f_name)
  positive_labels = np.ones(positive.shape[0])

  print 'Loading negative samples...'
  negative = load_ep_pairs(neg_enhancer_f_name, neg_promoter_f_name)
  negative_labels = np.zeros(negative.shape[0])

  samples = np.concatenate((positive, negative), 0)
  labels = np.concatenate((positive_labels, negative_labels), 0)
  return samples, labels

# TODO!!
def load_imbalanced_data(pos_neg_ratio, pos_enhancer_f_name, pos_promoter_f_name, neg_enhancer_f_name, neg_promoter_f_name):

  print 'Loading positive samples...'
  positive = load_ep_pairs(pos_enhancer_f_name, pos_promoter_f_name)
  n_pos = positive.shape[0] # original number of positive samples
  n_pos_sub = round(n_pos * pos_neg_ratio) # number of subsampled positive samples

  positive_subsampled = positive[np.random.choice(n_pos, n_pos_sub, replace=False),:, :]
  positive_labels = np.ones(n_pos_sub)

  print 'Loading negative samples...'
  negative = load_ep_pairs(neg_enhancer_f_name, neg_promoter_f_name)
  negative_labels = np.zeros(negative.shape[0])

  samples = np.concatenate((positive_subsampled, negative), 0)
  labels = np.concatenate((positive_labels, negative_labels), 0)
  return samples, labels

# Reads the specified files and builds num_samples X 4 X seg_length arrays for
# both the enhancers and promoter data sets; returns concatenations of enhancer
# and promoter segments
def load_ep_pairs(enhancer_f_name, promoter_f_name):

  print 'Loading enhancer data from ' + enhancer_f_name + '...'
  enhancers = load_file(enhancer_f_name)

  # # Code to just load enhancers
  # print 'Not using any promoters at the moment!'
  # promoters = np.zeros((enhancers.shape[0], 4, 0))

  print 'Loading promoter data from ' + promoter_f_name + '...'
  promoters = load_file(promoter_f_name)

  return np.concatenate((enhancers, promoters), 2)

# Reads the specified files and builds num_samples X 4 X seg_length arrays for
# both the positive and negative data sets; returns concatenations of positive
# and negative data sets, as well as the labels
def load_pn_data(positive_f_name, negative_f_name):

  print 'Loading positive data from ' + positive_f_name + '...'
  positive = load_file(positive_f_name)
  positive_labels = np.ones(positive.shape[0])

  print 'Loading negative data from ' + negative_f_name + '...'
  negative = load_file(negative_f_name)
  negative_labels = np.zeros(negative.shape[0])

  samples = np.concatenate((positive, negative), 0)
  labels = np.concatenate((positive_labels, negative_labels), 0)
  return samples, labels

# Reads the specified file and returns a 3D
# num_samples X 4 X segment_length array
def load_file(f_name):

  num_pairs = sum(1 for line in open(f_name))/4; # number of sample pairs

  # Declare output var but can't allocate space till we know segment_length
  Xs = None

  with open(f_name, 'r') as f:

    line_num = 0 # number of lines (i.e., samples) read so far
    for line in f.read().splitlines():

      if line_num == 0:
        # allocate space for output
        Xs = np.zeros((num_pairs, 4, len(line)))

      sample_type = line_num % 4; # 0, 1, 2, or 4, denoting A, T, C, or G
      sample_num = line_num / 4;

      Xs[sample_num, sample_type, :] = [int(x) for x in line];

      line_num += 1

  return Xs
