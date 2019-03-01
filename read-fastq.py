from __future__ import division
from Bio import SeqIO
import numpy as np
import tqdm
from nltk.metrics import edit_distance
from fast5_research import Fast5
import matplotlib.pyplot as plt
#############################################################################################################################3
def  
## "accuracy" 
  use_modified_chiron = 1
  levenstein_dis = 1
  
  filename = "basecalled-fastQ/hlapoolbc1.fastq"
  raw_filename = ""#raw-fast5/NANOPORE_20161222_FNFAB48816_MN19414_mux_scan_161222_hla_pool_8bc_29898_ch124_read12_strand.fast5"
  #records = list(SeqIO.parse(filename,"fastq"))
  #print(len(records))# 51354 records
  count = -1
  for record in SeqIO.parse(filename,'fastq'):
    count = count + 1
    if(count==6):
      print("reading the seq")
      base_seq = record.seq
    elif(count==7):
      base_seq = base_seq +record.seq
      print('reading the complement')
    elif(count==8):
      template_deq = record.seq
      print('finished from NANOPore')
      break
    
  filename_chiron = "chiron-result-five/result/NANOPORE_20161222_FNFAB48816_MN19414_mux_scan_161222_hla_pool_8bc_29898_ch130_read80_strand.fastq"
  #"chiron-result/result/NANOPORE_20161222_FNFAB48816_MN19414_mux_scan_161222_hla_pool_8bc_29898_ch124_read24_strand.fastq"
  for record in SeqIO.parse(filename_chiron,'fastq'):
    base_seq_chiron = record.seq
    print('succesfully read from Chiron')
  
  if use_modified_chiron:
    filename_chiron_mod = "chiron-result-mod/result/NANOPORE_20161222_FNFAB48816_MN19414_mux_scan_161222_hla_pool_8bc_29898_ch130_read80_strand.fastq"
    #"chiron-result/result/NANOPORE_20161222_FNFAB48816_MN19414_mux_scan_161222_hla_pool_8bc_29898_ch124_read24_strand.fastq"
    for record in SeqIO.parse(filename_chiron_mod,'fastq'):
      base_seq_chiron_mod = record.seq
      print('succesfully read from Chiron-modified')
  if levenstein_dis:
    print(str(edit_distance(base_seq,base_seq_chiron)))#2154
    print(str(edit_distance(base_seq,base_seq_chiron_mod)))#2603
    print(str(edit_distance(base_seq_chiron_mod,base_seq_chiron)))#2596

manual_match = 0
if manual_match:
  seg_size = 12
  #########################################################
  search_r = len(base_seq_chiron)-seg_size 
  neighb = 1000
  ali_idx_l = np.zeros(search_r)
  ali_idx_m = np.zeros(search_r)
  ali_idx_h = np.zeros(search_r)
  print("finding the matches between nanopore and chiron_default...")
  for i in tqdm.trange(search_r):
    segment = base_seq_chiron[i:i+seg_size]
    for j in range(len(base_seq)-seg_size):#range(max(0,i-neighb),min(len(base_seq)-seg_size,i+neighb)):
      count = 0
      for k in range(seg_size):
        if(segment[k] == base_seq[j+k]):
          count = count + 1
      count = count/seg_size
      if(count==1):
        ali_idx_h[i] = j
      elif(count > 0.91):
        ali_idx_m[i] = j
      #elif(count>0.83):
        #ali_idx_l[i] = j
      #if(segment == base_seq[j:j+seg_size]):
      #  ali_idx[i] = j
      #  break
  print("find full match : "+str(np.count_nonzero(ali_idx_h))+' find middle match : '+str(np.count_nonzero(ali_idx_m))+' find low match(default to ignore) '+str(np.count_nonzero(ali_idx_l)))
  nzl = np.nonzero(ali_idx_l)[0] #1113 / 1479
  nzm = np.nonzero(ali_idx_m)[0] #1197 / 1687
  nzh = np.nonzero(ali_idx_h)[0] # 0
  plt.figure(1)
  plt.plot(nzl,ali_idx_l[nzl],'g+',nzm,ali_idx_m[nzm],'b+',nzh,ali_idx_h[nzh],'r+')
  plt.plot(range(8000),range(8000),'b--')
  plt.grid(True)
  #################################################################
  search_r = len(base_seq_chiron_mod)-seg_size 
  ali_idx_l = np.zeros(search_r)
  ali_idx_m = np.zeros(search_r)
  ali_idx_h = np.zeros(search_r)
  print("finding the matches between nanopore and chiron_modified...")
  for i in tqdm.trange(search_r):
    segment = base_seq_chiron_mod[i:i+seg_size]
    for j in range(len(base_seq)-seg_size):#range(max(0,i-neighb),min(len(base_seq)-seg_size,i+neighb)):
      count = 0
      for k in range(seg_size):
        if(segment[k] == base_seq[j+k]):
          count = count + 1
      count = count/seg_size
      if(count==1):
        ali_idx_h[i] = j
      elif(count > 0.91):
        ali_idx_m[i] = j
      #elif(count>0.83):
        #ali_idx_l[i] = j
      #if(segment == base_seq[j:j+seg_size]):
      #  ali_idx[i] = j
      #  break
  print("find full match : "+str(np.count_nonzero(ali_idx_h))+' find middle match : '+str(np.count_nonzero(ali_idx_m))+' find low match(default to ignore) '+str(np.count_nonzero(ali_idx_l)))
  nzl = np.nonzero(ali_idx_l)[0] #485 / 448
  nzm = np.nonzero(ali_idx_m)[0] #793 / 1015
  nzh = np.nonzero(ali_idx_h)[0]
  plt.figure(2)
  plt.plot(range(8000),range(8000),'b--')
  plt.plot(nzl,ali_idx_l[nzl],'g+',nzm,ali_idx_m[nzm],'b+',nzh,ali_idx_h[nzh],'r+')
  plt.grid(True)
  ################################################################
  search_r = len(base_seq_chiron)-seg_size 
  ali_idx_l = np.zeros(search_r)
  ali_idx_m = np.zeros(search_r)
  ali_idx_h = np.zeros(search_r)
  print("finding the matches between chiron_modified and chiron_modified...")
  for i in tqdm.trange(search_r):
    segment = base_seq_chiron[i:i+seg_size]
    for j in range(len(base_seq_chiron_mod)-seg_size):#range(max(0,i-neighb),min(len(base_seq)-seg_size,i+neighb)):
      count = 0
      for k in range(seg_size):
        if(segment[k] == base_seq_chiron_mod[j+k]):
          count = count + 1
      count = count/seg_size
      if(count==1):
        ali_idx_h[i] = j
      elif(count > 0.91):
        ali_idx_m[i] = j
      #elif(count>0.83):
        #ali_idx_l[i] = j
      #if(segment == base_seq[j:j+seg_size]):
      #  ali_idx[i] = j
      #  break
  print("find full match : "+str(np.count_nonzero(ali_idx_h))+' find middle match : '+str(np.count_nonzero(ali_idx_m))+' find low match(default to ignore) '+str(np.count_nonzero(ali_idx_l)))
  nzl = np.nonzero(ali_idx_l)[0] #856 / 1200
  nzm = np.nonzero(ali_idx_m)[0] #1148 / 1656
  nzh = np.nonzero(ali_idx_h)[0]  
  plt.figure(3)
  plt.plot(range(8000),range(8000),'b--')
  plt.plot(nzl,ali_idx_l[nzl],'g+',nzm,ali_idx_m[nzm],'b+',nzh,ali_idx_h[nzh],'r+')
  plt.grid(True)

  
  plt.draw()
  plt.pause(1) # <-------
  raw_input("<Hit Enter To Close>")
  plt.close(plt.figure())
#############################################################################################################################
##########################################################################################################################3
## Find the raw fast5 and the correspoding fastq
Rnum = 0
raw_filename = "something"
count = 1
#####################
# flag for finding the pair of fast5 and fastq
fast5_search =1
#####################
if fast5_search:
  for record in SeqIO.parse(filename,"fastq"):
    if(Rnum < 50):
      narv = record.description
      id_start = narv.find('/NANOPORE_')
      id_end = len(narv)
      id_raw = narv[id_start:id_end]
      #print(len(record.letter_annotations['phred_quality']))
      Rnum = Rnum + 1
  
      raw_filename_ = "raw-fast5"+ id_raw
      #print(raw_filename_)
      #try:
      with Fast5(raw_filename_) as fh:
        
        count = count + 1
        print(count)
        if(raw_filename!=raw_filename_):
          print("find new raw source"+raw_filename_)
          count = 1
          raw_filename = raw_filename_
          raw = fh.get_read(raw=True)
          base_seq = record.seq
        elif(count == 2):
          print('reading complement')
          base_seq = base_seq+record.seq[::-1]
        elif(count == 3):
          print('reading template')
          template = record.seq
          plt.subplot(2,1,1)
          plt.plot(raw)
          plt.subplot(2,1,2)
          plt.plot(base_seq)
          plt.draw()
          plt.pause(1) # <-------
          raw_input("<Hit Enter To Close>")
          plt.close(plt.figure())
      #except IOError:
       # pass
    else:
      break
  
