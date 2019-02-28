import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from fast5_research import Fast5
from Bio import SeqIO
np.set_printoptions(threshold=sys.maxsize)

class Params(object):
  def __init__(self):
    self.Kmer = 4
    self.write_file = False
    self.step = 1
    self.length = 300
    self.extra_testdata = False
    if len(sys.argv)>3:
      print('project only takes maximal 3 inputs!')
      sys.exit(0)
    self.paths_a = sys.argv[1:3]
    self.paths_b = []
    if len(sys.argv)>3:
      self.extra_testdata = True
      self.paths_b = list(sys.argv[i] for i in [1,3])
PARAMS = Params()

def fetch_HMM_fn(paths):
  mypath_raw= paths[0]
  mypath_data = paths[1]
  for file in os.listdir(mypath_raw):
    # generate the three kinds of file name strings
    fast5_fn = os.fsdecode(file)
    result_chiron_fn = mypath_data+'/result/'+fast5_fn[:-5]+'fastq' # the base sequence
    raw_fn = mypath_raw +'/'+fast5_fn # raw reads/inputs file
    # generate three data objects iterator
    for record in SeqIO.parse(result_chiron_fn,'fastq'):
      base_seq = record.seq # only one sequence per file
    raw_fn = Fast5(raw_fn)
    raw = raw_fn.get_read(raw=True)
    yield fast5_fn,base_seq, raw

def fetch_classif_fn(paths)
  mypath_raw= paths[0]
  mypath_data = paths[1]
  for file in os.listdir(mypath_raw):
    # generate the three kinds of file name strings
    fast5_fn = os.fsdecode(file)
    result_chiron_fn = mypath_data+'/result/'+fast5_fn[:-5]+'fastq' # the base sequence
    base_seqidx_fn = mypath_data+'/'+fast5_fn[:-5] + 'signalsegidx.txt' # the seq_idx file for interaction
    raw_fn = mypath_raw +'/'+fast5_fn # raw reads/inputs file
    # generate three data objects iterator
    for record in SeqIO.parse(result_chiron_fn,'fastq'):
      base_seq = record.seq # only one sequence per file
    f_seqidx = open(base_seqidx_fn,'r')
    raw_fn = Fast5(raw_fn)
    raw = raw_fn.get_read(raw=True)
    yield fast5_fn,base_seq, f_seqidx, raw



def classif_data():
  base_k_list = list()
  raw_k_list = list()
  fn_iter = fetch_classif_fn(PARAMS.paths_a)
  for fn,base_seq, f_seqidx, raw in fn_iter:
    print(fn+'has '+str(len(raw))+' raw data')
    old_line = np.array([])
    intsec = list()
    count = 0
    for line in f_seqidx:
      count += 1
      a_line = np.asarray(line.strip().split(' '))
      a_line = a_line.astype(int)
      if len(old_line)==0:
        old_line = a_line
        continue
      for it in range(k-3,-1,-1):
        if len(intsec)>=it+1:
          if len(intsec)<it+2:
            intsec.append(np.intersect1d(intsec[it],a_line))
          else:
            intsec[it+1] = np.intersect1d(intsec[it],a_line)
      if len(intsec)==k-1:
        if len(intsec[-1])!=0:
          num_intsec = len(intsec[-1])
          indx = int(sorted(intsec[-1])[num_intsec//2-1]-1) #-1 since index is writen starts from 1
          if count > 20:
            if len(raw[indx*30:indx*30+300])==300:
              raw_k_list.append(raw[indx*30:indx*30+300])
              base_k_list.append(base_seq[count-k:count])
              if PARAMS.write_file:
                #myfile_input.write(' '.join(map(str,raw[indx*30:indx*30+300])))
                myfile_input.write(str(intsec)+' '+str(indx))
                myfile_input.write('\n')
                myfile_output.write(' '.join(base_seq[count-k:count]))
                myfile_output.write('\n')
      if len(intsec)==0:
        intsec.append(np.intersect1d(old_line,a_line))
        #print(str(intsec))
      else:
        intsec[0] = np.intersect1d(old_line,a_line)
        #print(str(intsec))
      old_line = a_line
    f_seqidx.close()
  return raw_k_list, base_k_list

def HMM_data():
  # data for HMM
  #self.raw_list ~ collection of raw reads
  #self.base_list ~ collection of DNA chains
  #self.base_o_list ~ k-mer list for P_o
  #self.base_seqk_list ~ K-mer and its follower list for transition matrix
  base_list = list()
  raw_list = list()
  base_o_list = list()
  base_seqk_list = list()
  fn_iter = fetch_HMM_fn(PARAMS.paths_a)
  for fn,base_seq, raw in fn_iter:
    print(fn+'has '+str(len(raw))+' raw data')
    base_list.append(base_seq)
    ##############################
    # get one-step skip raw data windows for HMM
    raw_len = len(raw)
    raw_step_list = list()
    for i in range(0,raw_len,PARAMS.step):
      if len(raw[i:i+PARAMS.length]) == PARAMS.length:
        raw_step_list.append(raw[i:i+PARAMS.length])
    raw_list.append(np.asarray(raw_step_list))
    ###############################
    # get pair wise DNA kmer base for HMM
    base_a = None
    base_b = np.array([])
    base_len = len(base_seq)
    for i in range(base_len-k):
      base_a = ''.join(base_seq[i:i+k])
      base_o_list.append(base_a)
      if len(base_b)>0:
        base_seqk_list.append(np.stack((base_b,base_a)))
      base_b = base_a
  return raw_list, base_list, base_o_list, base_seqk_list

class DNAData(object):
  def __init__(self, PARAMS):
    self.k = PARAMS.Kmer
  ###################################################################################
  # get intersection and its corresponding raw data for classification training  
  def get_HMM_data(self):
    if PARAMS.write_file:
      input_fn = 'data-training/input_data.txt'
      output_fn = 'data-training/output_data.txt'
      myfile_input = open(input_fn,"w")
      myfile_output = open(output_fn,"w")
    self.raw_list, self.base_list, self.base_o_list, self.base_seqk_list = HMM_data()
  ####################################################################################
  # data for classification
  def get_classif_data(self):
    raw_k_list, base_k_list = classif_data()
    self.output_table, self.output_basenum = DNAData.base2table(base_k_list,self.k) # K-mer list with number(k^4) instead of letter 
    #self.output_vec = DNAData.base2vec(base_k_list,k) # K-mer list with number(k*4) instead of letter
    self.input_table = np.stack(raw_k_list)
    self.raw_k_list = raw_k_list # raw read per 300 w.r.t base

########################################################################################
# get another group of test data
    if PARAMS.extra_testdata:
      base_list = list()
      raw_list = list()
      raw_k_list = list()
      base_o_list = list()
      base_k_list = list()
      base_seqk_list = list()
      intsec = list()      
      fn_iter = fetch_fn(PARAMS.paths_b)
      print('getting another data set')
      for fn,base_seq, f_seqidx, raw in fn_iter:
        print(fn+'has '+str(len(raw))+' raw data')
        ############################
        # get the DNA base for HMM
        base_list.append(base_seq)
        old_line = np.array([])
        intsec = list()
        ##############################
        # get one-step skip raw data for HMM
        raw_step_list = list()
        raw_len = len(raw)
        for i in range(0,raw_len,step):
          if len(raw[i:i+length]) == length:
            raw_step_list.append(raw[i:i+length])
        raw_list.append(np.asarray(raw_step_list))
        ###############################
        # get pair wise DNA kmer base for HMM
        base_a = None
        base_b = np.array([])
        base_len = len(base_seq)
        for i in range(base_len-k):
          base_a = ''.join(base_seq[i:i+k])
          base_o_list.append(base_a)
          if len(base_b)>0:
            base_seqk_list.append(np.stack((base_b,base_a)))
          base_b = base_a
        ################################
        # get intersection and its corresponding raw data for classification training
        count = 0
        for line in f_seqidx:
          count += 1
          a_line = np.asarray(line.strip().split(' '))
          a_line = a_line.astype(int)
          if len(old_line)==0:
            old_line = a_line
            continue
          for it in range(k-3,-1,-1):
            if len(intsec)>=it+1:
              if len(intsec)<it+2:
                intsec.append(np.intersect1d(intsec[it],a_line))
                #print(str(intsec))
              else:
                intsec[it+1] = np.intersect1d(intsec[it],a_line)
                #print(str(intsec))
          if len(intsec)==k-1:
            if len(intsec[-1])!=0:
              #print(str(intsec[k-2]))
              num_intsec = len(intsec[-1])
              indx = int(sorted(intsec[-1])[num_intsec//2-1]-1) #-1 since index is writen starts from 1
              if count > 20:
                if len(raw[indx*30:indx*30+300])==300:
                  raw_k_list.append(raw[indx*30:indx*30+300])
                  base_k_list.append(base_seq[count-k:count])
                  if write_file:
                    #myfile_input.write(' '.join(map(str,raw[indx*30:indx*30+300])))
                    myfile_input.write(str(intsec)+' '+str(indx))
                    myfile_input.write('\n')
                    myfile_output.write(' '.join(base_seq[count-k:count]))
                    myfile_output.write('\n')
          if len(intsec)==0:
            intsec.append(np.intersect1d(old_line,a_line))
            #print(str(intsec))
          else:
            intsec[0] = np.intersect1d(old_line,a_line)
            #print(str(intsec))
          old_line = a_line      

    ####################################################################################
    # data for HMM
    self.raw_list2 = raw_list # collection of raw reads
    self.base_list2 = base_list # collection of DNA chains
    self.base_o_list2 = base_o_list # k-mer list for P_o
    self.base_seqk_list2 = base_seqk_list # K-mer and its follower list for transition matrix
    ####################################################################################
    # data for classification (intersections from chiron)
    self.output_table2,self.output_basenum2 = DNAData.base2table(base_k_list,k) # K-mer list with number(k^4) instead of letter 
    self.output_vec2 = DNAData.base2vec(base_k_list,k) # K-mer list with number(k*4) instead of letter
    self.input_table2 = np.stack(raw_k_list)
    self.raw_k_list2 = raw_k_list # raw read per 300 w.r.t base
    if write_file:  
      myfile_input.close()
      myfile_output.close()
  
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  #################################################3
  def check_balance(self):
    kmer,dup = np.unique(self.output_basenum,return_counts=True)
    mostk_num = np.argmax(dup)
    mostk = DNAData.base2num([mostk_num],self.k,op=True)
    dup = dup/np.sum(dup)
    kmer2,dup2 = np.unique(self.output_basenum2,return_counts=True)
    mostk_num2 = np.argmax(dup2)
    mostk2 = DNAData.base2num([mostk_num2],self.k,op=True)
    dup2 = dup2/np.sum(dup2)
    
    print('the train data has class: \n {'+str(kmer)+"} \n with each partation:\n "+str(dup) +' \n The most frequent kner is '+str(mostk_num)+' '+str(mostk))
    print('the test data has class: \n {'+str(kmer2)+"} \n with each partation:\n "+str(dup2)+' \n The most frequent kner is '+str(mostk_num2)+' '+str(mostk2) )
  ##################################################3
  def get_data(self):
    output_set = self.output_table
    input_set = self.input_table
    num_data,_ = np.shape(input_set)
    indx = np.arange(num_data)
    np.random.shuffle(indx)
    
    input_test = input_set[indx[(num_data//5)*4:num_data-1],:]
    output_test = output_set[indx[(num_data//5)*4:num_data-1],:]
    input_train = input_set[indx[0:(num_data//5)*4],:]
    output_train = output_set[indx[0:(num_data//5)*4],:] 
    return input_train, output_train, input_test, output_test
  def get_another_data(self):
    return self.input_table2,self.output_table2 
  
  
  def get_ZX(self):
    Z_o = self.base_o_list
    Z_trans = self.base_seqk_list
    numz = len(Z_trans)
    Z_trans = Z_trans[0:numz//2]
    X_test = self.raw_list
    Z_test = self.base_list
    return Z_trans, X_test, Z_test, Z_o
  def get_another_ZX(self):
    X_test = self.raw_list2
    Z_test = self.base_list2
    return X_test, Z_test
