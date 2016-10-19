import numpy as np
import math
import matplotlib.pyplot as plt
import build_model
from scipy.cluster import vq

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = build_model.build_model(use_JASPAR = False)

cell_line = 'K562'
model.load_weights('/home/sss1/Desktop/projects/DeepInteractions/weights/balanced' + cell_line + '-noJASPAR-2016-10-13-18:40:51.hdf5')

enhancer_conv_weights = np.squeeze(model.layers[0].layers[0].layers[0].get_weights()[0])
# # Just as a test, generate weights according to random noise
# enhancer_conv_weights = np.random.normal(size = (40, 4, 1024))
promoter_conv_weights = np.squeeze(model.layers[0].layers[1].layers[0].get_weights()[0])

# These each have shape (40, 4, 1024)

# Min correction: add min to each row and then normalize to sum to 1
# Non-negative correction: Zero all negative entries and normalize to sum to 1
print 'MEME version 4\n'
print 'ALPHABET= ACGT\n'
print 'strands: + -\n'
print 'Background letter frequencies (from uniform background):'
print 'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n'

def print_motifs(weight_tensor, EP):
  kernel_length, _, num_kernels = np.shape(weight_tensor)
  motif_spec = 'letter-probability matrix: alength= 4 w= ' + str(kernel_length)
  for kernel_idx in range(2):#num_kernels):
    print '\nMOTIF ' + cell_line + '-' + EP + ("%04d" % kernel_idx)
    print motif_spec
  
    for position in range(kernel_length):
      row = weight_tensor[position, :, kernel_idx]
      row = np.maximum(row, 0)# -= row.min()
      row /= row.sum()
      if np.isnan(row).any(): # if all weights in row are negative, weight the bases uniformly
        row = [0.25, 0.25, 0.25, 0.25]
      print "%.6f  %.6f  %.6f  %.6f" % (row[0], row[2], row[1], row[3])

print_motifs(enhancer_conv_weights, 'E')
print_motifs(promoter_conv_weights, 'P')

# enhancer_all = enhancer_conv_weights.reshape((40 * 4, 1024)).transpose([1,0])
# centroids, variance  = vq.kmeans(enhancer_all, 3)
# identified, distance = vq.vq(enhancer_all, centroids)
# 
# # for i in range(100):#enhancer_all.shape(0)):
# #   if identified[i] == 0:
# #     color = 'r'
# #   elif identified[i] == 1:
# #     color = 'g'
# #   else:
# #     color = 'b'
# # 
# #   plt.plot(enhancer_all[i, :], color = color)
# # 
# # plt.show()
# 
# X = enhancer_all
# y = identified
# target_names = ['0', '1', '2']
# 
# pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
# 
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(X, y).transform(X)
# 
# # Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s'
#       % str(pca.explained_variance_ratio_))
# 
# plt.figure()
# colors = ['navy', 'turquoise', 'darkorange']
# lw = 2
# 
# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('PCA')
# 
# plt.figure()
# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA')
# 
# plt.show()
