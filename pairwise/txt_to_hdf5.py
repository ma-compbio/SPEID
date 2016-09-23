# Basic python and data processing imports
import numpy as np
import h5py
from sklearn.utils import shuffle
import load_data_pairs as ld # my own scripts for loading data

# input data paths
data_dir = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/K562/'
positive_enhancers = data_dir + 'K562_pos_1.txt'
positive_promoters = data_dir + 'K562_pos_2.txt'
negative_enhancers = data_dir + 'K562_neg_1.txt'
negative_promoters = data_dir + 'K562_neg_2.txt'

# output data path
out_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/K562/K562.h5'

print 'Loading data sets...'
X_train, y_train = ld.load_full_data(positive_enhancers,
                                        positive_promoters,
                                        negative_enhancers,
                                        negative_promoters)
X_train = np.transpose(X_train, axes=(0,2,1))

# shuffle the data (since keras does not shuffle validation data)
print 'Shuffling data...'
rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(y_train)

print 'saving data to ' + out_path + '...'
with h5py.File(out_path, 'w') as hf:
    hf.create_dataset('X_train', data = X_train)
    hf.create_dataset('y_train', data = y_train)
