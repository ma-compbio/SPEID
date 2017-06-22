# Basic python and data processing imports
import numpy as np
import h5py
from sklearn.utils import shuffle
import load_data_pairs as ld # my own scripts for loading data

# input data paths
cell_line = '4lines'
data_prefix = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/' + cell_line + '/' + cell_line
positive_enhancers = data_prefix + '_pos_1.txt'
positive_promoters = data_prefix + '_pos_2.txt'
positive_enhancers_aug = data_prefix + '_pos_1_aug.txt'
positive_promoters_aug = data_prefix + '_pos_2_aug.txt'
negative_enhancers = data_prefix + '_neg_1.txt'
negative_promoters = data_prefix + '_neg_2.txt'

# Save state of random number generator so we can jointly shuffle data
rng_state = np.random.get_state()

# output data path
out_path = data_prefix + '_ep_split.h5'

with h5py.File(out_path + 'ori_enh', 'w') as hf:

    print 'Loading original enhancers...'
    X_enhancers_train, y_train = ld.load_pn_data(positive_enhancers, negative_enhancers)
    X_enhancers_train = np.transpose(X_enhancers_train, axes=(0,2,1))
    print 'Shuffling original enhancers...'
    np.random.set_state(rng_state)
    np.random.shuffle(X_enhancers_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)
    print 'Saving original enhancers...'
    hf.create_dataset('X_enhancers_train', data = X_enhancers_train)
    hf.create_dataset('y_train', data = y_train)

with h5py.File(out_path + 'ori_pro', 'w') as hf:

    print 'Loading original promoters...'
    X_promoters_train, _ = ld.load_pn_data(positive_promoters, negative_promoters)
    X_promoters_train = np.transpose(X_promoters_train, axes=(0,2,1))
    print 'Shuffling original promoters...'
    np.random.set_state(rng_state)
    np.random.shuffle(X_promoters_train)
    print 'Saving original promoters...'
    hf.create_dataset('X_promoters_train', data = X_promoters_train)

with h5py.File(out_path + 'aug_enh', 'w') as hf:

    print 'Loading augmented enhancers...'
    X_enhancers_train_aug, y_train_aug = ld.load_pn_data(positive_enhancers_aug, negative_enhancers)
    X_enhancers_train_aug = np.transpose(X_enhancers_train_aug, axes=(0,2,1))
    print 'Shuffling augmented enhancers...'
    np.random.set_state(rng_state)
    np.random.shuffle(X_enhancers_train_aug)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train_aug)
    print 'Saving augmented enhancers...'
    hf.create_dataset('X_enhancers_train_aug', data = X_enhancers_train_aug)
    hf.create_dataset('y_train_aug', data = y_train_aug)

with h5py.File(out_path + 'aug_pro', 'w') as hf:

    print 'Loading augmented promoters...'
    X_promoters_train_aug, _ = ld.load_pn_data(positive_promoters_aug, negative_promoters)
    X_promoters_train_aug = np.transpose(X_promoters_train_aug, axes=(0,2,1))
    print 'Shuffling augmented promoters...'
    np.random.set_state(rng_state)
    np.random.shuffle(X_promoters_train_aug)
    print 'Saving augmented promoters...'
    hf.create_dataset('X_promoters_train_aug', data = X_promoters_train_aug)
