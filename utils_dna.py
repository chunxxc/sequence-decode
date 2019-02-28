import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def base2table(bases,k=4):
  # from ACTG to one-hot-coding table of 0~4^k
  base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
  num_kmer = len(bases)
  output_idx = np.zeros((num_kmer,4**k))
  output_basenum = np.zeros(num_kmer)
  for j in range(num_kmer):
    kmer = bases[j]
    #print(kmer)
    indx = 0
    for i in range(k-1,-1,-1):
      indx += base_dict[kmer[k-1-i]]*4**i
    output_idx[j,int(indx)] = 1
    output_basenum[j] = indx
  return output_idx.astype(int), output_basenum  

def base2vec(bases,k=4):
  # from ACTG to one-hot-coding of 0~4*k
  base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
  num_kmer = len(bases)
  output_vec =np.zeros((num_kmer,k*4))
  for i in range(num_kmer):
    for j in range(k):
      output_vec[i,j*4+base_dict[bases[i][j]]] = int(1)
  #output_vec = int(output_vec)
  return output_vec.astype(int)

def base2num(Z, K, op=False):
  # for transfer base to indx
  base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
  base_dict_op = {0:'A', 1:'C', 2:'G', 3:'T'}
  T = len(Z)
  if not op:
    Z_ = np.zeros(T)  
    for i in range(T):
      kmer = Z[i]
      indx = 0
      for j in range(K-1,-1,-1):
        indx += base_dict[kmer[K-1-j]]*(4**j)
      Z_[i] = indx
    Z_ = Z_.astype(int)
  else:
    Z_ = list()
    for i in range(T):
      kmer = Z[i]
      base = None
      for j in range(K-1,-1,-1):
        indx = kmer//(4**j)
        kmer = kmer%(4**j)
        if not base:
          base = base_dict_op[indx]
          continue
        base += base_dict_op[indx] 
      Z_.append(base)
  return Z_
  

