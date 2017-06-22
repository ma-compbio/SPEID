import numpy as np
import load_data_pairs as ld
import h5py

root = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/original/'
data_path = root + 'all_data.h5'
cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']

# Map:
#   [1,0,0,0] -> A
#   [0,1,0,0] -> T
#   [0,0,1,0] -> C
#   [0,0,0,1] -> G
def one_hot_to_letter(base):
      if not (np.sum(base) == 1 and np.min(base) == 0):
        return 'N'
      if base[0] == 1:
        return 'A'
      if base[1] == 1:
        return 'T'
      if base[2] == 1:
        return 'C'
      if base[3] == 1:
          return 'G'
      return 'N'

# data: (num_sequences X sequence_length X 4) 3-tensor of num_sequences
#   one-hot encoded nucleotide sequences of equal-length sequence_length
# name: string label for the data set (e.g., 'K562_enhancers')
# path: string file path to which to print the data
def format_file(data, name):
  print 'Formatting ' + name + ' data ...'
  file_to_print = ''
  sequence_idx = 0
  for sequence in data:
    sequence_to_print = ''
    for base in sequence:
      sequence_to_print += str(one_hot_to_letter(base))

    file_to_print += '>' + str(sequence_idx) + '\n'
    file_to_print += sequence_to_print + '\n'
    sequence_idx += 1

  return file_to_print


with h5py.File(data_path, 'r') as hf:

  for cell_line in cell_lines:
    print 'Loading ' + cell_line + ' data from ' + data_path
  
    # Print enhancer data
    X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
    name = cell_line + '_enhancers'
    out_path = root + 'asFASTA/' + name + '.fasta'
    file_contents = format_file(X_enhancers, name)
    print 'Writing ' + name + ' data to ' + out_path + ' ...'
    f = open(out_path, 'w')
    f.write(file_contents)
    f.close()
  
    # Print promoter data
    X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
    name = cell_line + '_promoters'
    out_path = root + 'asFASTA/' + name + '.fasta'
    file_contents = format_file(X_promoters, name)
    print 'Writing ' + name + ' data to ' + out_path + ' ...'
    f = open(out_path, 'w')
    f.write(file_contents)
    f.close()
