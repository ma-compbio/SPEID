# Basic python and data processing imports
import numpy as np
import h5py
from sklearn.utils import shuffle
import load_data_pairs as ld # my own scripts for loading data

# input data paths
combined_name = '4lines'
data_root = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/'

# output data path
out_path = data_root + combined_name + '/' + combined_name + '_ep_split.h5'

source_prefixes = ['GM12878', 'HeLa-S3', 'IMR90', 'K562']

# Sizes of fixed data dimensions
enhancers_length = 3000
promoters_length = 2000
num_bases = 4

X_enhancers_train_combined = np.zeros((0, enhancers_length, num_bases))
X_promoters_train_combined = np.zeros((0, promoters_length, num_bases))
y_train_combined = np.zeros((0,))
X_enhancers_train_aug_combined = np.zeros((0, enhancers_length, num_bases))
X_promoters_train_aug_combined = np.zeros((0, promoters_length, num_bases))
y_train_aug_combined = np.zeros((0,))

for prefix in source_prefixes:

  source_path = data_root + prefix + '/' + prefix + '_ep_split.h5'
  print 'Loading ' + prefix + ' data...'
  X_enhancers_train, X_promoters_train, y_train = ld.load_hdf5_ep_split(source_path)
  X_enhancers_train_aug, X_promoters_train_aug, y_train_aug = ld.load_hdf5_ep_split_aug(source_path)

  print 'Concatenating ' + prefix + ' data...'
  X_enhancers_train_combined = np.concatenate((X_enhancers_train_combined, X_enhancers_train))
  X_promoters_train_combined = np.concatenate((X_promoters_train_combined, X_promoters_train))
  y_train_combined = np.concatenate((y_train_combined, y_train))
  X_enhancers_train_aug_combined = np.concatenate((X_enhancers_train_aug_combined, X_enhancers_train_aug))
  X_promoters_train_aug_combined = np.concatenate((X_promoters_train_aug_combined, X_promoters_train_aug))
  y_train_aug_combined = np.concatenate((y_train_aug_combined, y_train_aug))

# Save state of random number generator so we can jointly shuffle data
print 'Shuffling data...'
rng_state = np.random.get_state()
np.random.set_state(rng_state)
np.random.shuffle(X_enhancers_train_combined)
np.random.set_state(rng_state)
np.random.shuffle(y_train_combined)
np.random.set_state(rng_state)
np.random.shuffle(X_promoters_train_combined)
np.random.set_state(rng_state)
np.random.shuffle(X_enhancers_train_aug_combined)
np.random.set_state(rng_state)
np.random.shuffle(y_train_aug_combined)
np.random.set_state(rng_state)
np.random.shuffle(X_promoters_train_aug_combined)

print 'Writing data...'
with h5py.File(out_path, 'w') as hf:
  hf.create_dataset('X_enhancers_train', data = X_enhancers_train)
  hf.create_dataset('y_train', data = y_train)
  hf.create_dataset('X_promoters_train', data = X_promoters_train)
  hf.create_dataset('X_enhancers_train_aug', data = X_enhancers_train_aug)
  hf.create_dataset('y_train_aug', data = y_train_aug)
  hf.create_dataset('X_promoters_train_aug', data = X_promoters_train_aug)
