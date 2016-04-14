import numpy as np
import os
import scipy.io as si

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x-1] = 1
	return o_h

def load_variables(ntrain = 5000,ntest = 1269):
    a = si.loadmat('/sequence.mat')
    b1 = a['label']
    list_of_strings = b1.tolist()
    i,j = b1.shape
    temp = list()
    for k in range(i):
        temp.append(list_of_strings[k][0])
    labels = np.asarray(temp)
    b2 = a['seq']
    i,j = b2.shape
    temp = list()
    for k in range(i):
        temp.append(b2[k])
    seqs = np.asarray(temp)
    trX = seqs[0:ntrain]
    teX = seqs[ntrain:ntrain+ntest]
    trY = labels[0:ntrain]
    teY = labels[ntrain:ntrain+ntest]
    trY_onehot = one_hot(trY, 2)
    teY_onehot = one_hot(teY, 2)
    return trX,teX,trY_onehot,teY_onehot
	
trX,teX,trY_onehot,teY_onehot = load_variables(ntrain = 5000,ntest = 1269)
print trY_onehot.shape


