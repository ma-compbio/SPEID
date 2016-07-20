import numpy as np
import sys

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

# Reads the specified files and builds num_samples X 4 X seg_length arrays for
# both the enhancers and promoter data sets; returns concatenations enhancer
# and promoter segments
def load_ep_pairs(enhancer_f_name, promoter_f_name):

    enhancer_seg_len = 1000
    promoter_seg_len = 1000

    print 'Loading enhancer data from ' + enhancer_f_name + '...'
    enhancers = load_file(enhancer_f_name, enhancer_seg_len)
    print 'Loading promoter data from ' + promoter_f_name + '...'
    promoters = load_file(promoter_f_name, promoter_seg_len)

    return np.concatenate((enhancers, promoters), 2)

# Reads the specified file and returns a 3D
# num_samples X 4 X segment_length array
def load_file(f_name, segment_length):

    num_pairs = sum(1 for line in open(f_name))/4; # number of sample pairs

    Xs = np.zeros((num_pairs, 4, segment_length)) # allocate space for output
    
    with open(f_name, 'r') as f:

        line_num = 0 # number of lines (i.e., samples) read so far
        for line in f.read().splitlines():

            sample_type = line_num % 4; # 0, 1, 2, or 4, denoting A, T, C, or G
            sample_num = line_num / 4;

            Xs[sample_num, sample_type, :] = [int(x) for x in line];

            line_num += 1

    return Xs
