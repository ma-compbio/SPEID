import numpy as np
import csv
from sklearn.metrics import average_precision_score
from keras.optimizers import Adam # needed to compile prediction model
import h5py
import load_data_pairs as ld # my own scripts for loading data
import build_small_model as bm
import util

fimo_root = '/home/sss1/Desktop/projects/DeepInteractions/pairwise/FIMO/'
data_path = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/original/all_data.h5'

cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
data_types = ['enhancers', 'promoters']

# num_repeats = 5 # number of i.i.d. trials to run; too slow to do :(
random_window_length = 20 # number of bp to randomize at each feature occurence

# Randomize each appearance of the pattern pattern in the data
def randomize_window(sequence):
  for base_idx in range(np.shape(sequence)[0]):
    sequence[base_idx] = np.zeros(4)
    sequence[base_idx, np.random.randint(0, 4)] = 1

# Returns a deep copy of the data, with motif occurrences randomized out.
# A deep copy is made because this is much faster than reloading the data for
# every motif.
# data: (num_sequences X sequence_length X 4) 3-tensor of num_sequences
#   one-hot encoded nucleotide sequences of equal-length sequence_length
# motifs_idxs: list of (sample_idx, start_idx, stop_idx) triples
def replace_motifs_in_data(data, motif_idxs):
  data_copy = np.copy(data)
  for (sample_idx, motif_start, motif_stop) in idxs:
    mid = (motif_start + motif_stop)/2
    start = max(0, mid - (random_window_length/2))
    stop = min(np.shape(data)[1], start + random_window_length)
    randomize_window(data_copy[sample_idx, start:stop, :])
  return data_copy

for cell_line in cell_lines:
  for data_type in data_types:

    fimo_path = fimo_root + cell_line + '_' + data_type + '_all_retest/fimo.txt'
    # data_path = data_root + cell_line + '/' + cell_line + '_ep_split.h5'

    matches = dict() # dict mapping motif_names to lists of (sample_idx, start_idx, stop_idx) triples

    print 'Reading and processing FIMO output...'
    with open(fimo_path, 'rb') as csv_file:
      reader = csv.reader(csv_file, delimiter='\t')

      row_idx = -1
      for row in reader:
        row_idx += 1
        if row_idx == 0: # skip header row
          continue

        motif_name = row[0]
        if not motif_name in matches: # if this is the first match of that motif
          matches[motif_name] = []

        sample_idx = int(row[1])
        motif_start = int(row[2])
        motif_stop = int(row[3])
        matches[motif_name].append((sample_idx, motif_start, motif_stop))

    print 'Identified ' + str(len(matches)) + ' distinct motifs.'
  
    print 'Loading original data...'
    # X_enhancers_original, X_promoters_original, y = ld.load_hdf5_ep_split(data_path)
    with h5py.File(data_path, 'r') as hf:
      X_enhancers_original = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
      X_promoters_original = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
      y = np.array(hf.get(cell_line + 'labels'))
      print 'np.shape(X_enhancers_original): ' + str(np.shape(X_enhancers_original))
      print 'np.shape(X_promoters_original): ' + str(np.shape(X_promoters_original))
      print 'np.shape(y): ' + str(np.shape(y))

    model = bm.build_model(use_JASPAR = False)
    
    print 'Compiling model...'
    opt = Adam(lr = 1e-5)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = opt,
                  metrics = ["accuracy"])
    print 'Loading ' + cell_line + ' ' + data_type + ' model weights...'
    model.load_weights('/home/sss1/Desktop/projects/DeepInteractions/weights/' + cell_line + '-basic.hdf5')
    out_root = '/home/sss1/Desktop/projects/DeepInteractions/feature_importances/SPEID/from_HOCOMOCO_motifs/'
    out_path = out_root + cell_line + '_' + data_type + '_feature_importance.csv'

    print 'Running predictions on original data'
    y_score = model.predict([X_enhancers_original, X_promoters_original], batch_size = 100, verbose = 1)
    true_AUPR = average_precision_score(y, y_score)
    print 'True AUPR is ' + str(true_AUPR)
    true_MS = y_score.mean()
    print 'True MS is ' + str(true_MS)

    with open(out_path, 'wb') as csv_file:

      writer = csv.writer(csv_file, delimiter = ',')
      writer.writerow(['Motif Name', 'Motif Count', 'AUPR Difference', 'MS Difference'])
      for motif, idxs in matches.iteritems():
        print 'Randomizing ' + str(len(idxs)) + ' occurrences of motif ' + motif + ' in ' + cell_line + ' ' + data_type + '...'
        if data_type == 'enhancers':
          X_enhancers = replace_motifs_in_data(X_enhancers_original, idxs)
          X_promoters = X_promoters_original
        elif data_type == 'promoters':
          X_enhancers = X_enhancers_original
          X_promoters = replace_motifs_in_data(X_promoters_original, idxs)
        else:
          raise ValueError

        print 'Running predictions on motif ' + motif + '...'
        y_score = model.predict([X_enhancers, X_promoters], batch_size = 200, verbose = 1)
        AUPR = average_precision_score(y, y_score)
        print 'AUPR after removing motif ' + motif + ' was ' + str(AUPR) + '\n'
        MS = y_score.mean()
        print 'MS after removing motif ' + motif + ' was ' + str(MS) + '\n'

        writer.writerow([motif, str(len(idxs)), str(true_AUPR - AUPR), str(true_MS - MS)])
