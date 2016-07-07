import numpy as np
import random as rand
import warnings
import sys

max_len = sys.maxsize # reduce this to prevent out-of-memory errors during prediction

# reads the specified file and returns a num_samples X 4 X 1000 array
def load_and_format_data(f_name):

    num_pairs = min(max_len, sum(1 for line in open(f_name))/4); # number of sample pairs

    Xs = np.zeros((num_pairs, 4, 1000)) # allocate space for output
    
    with open(f_name, 'r') as f:
    
        line_num = 0 # number of non-metadata lines (i.e., samples) read so far
        for line in f.read().splitlines():

            sample_type = line_num % 4;
            sample_num = line_num / 4;

            if sample_num >= max_len:
                warnings.warn("The data are very large. Truncating at " +
                        str(max_len) + " samples to avoid memory errors.")
                break

            Xs[sample_num, sample_type, :] = [int(x) for x in line];

            line_num += 1

    return Xs
