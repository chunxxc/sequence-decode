import sys
import time
import tables
from tempfile import TemporaryFile
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from creat_data import DNAData
from utils_dna import base2num
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
#CHECKPOINT_PATH = ./checkpoints/
###########################################################################################################
class HMM_DNA:  
  def A_init(K):
    base_dict_op = {0:'A', 1:'C', 2:'G', 3:'T'}
    A = np.zeros((4**K,4**K))
    for i in range(4**K):
      base = HMM_DNA.base2num([i],K,op=True)
      for j in range(4):
        indx = HMM_DNA.base2num([base[0][-K+1:]+base_dict_op[j]],K)
        A[i,indx] = 0.25
    return A
  # calculate prior of X based on MLE
  def prior_init(Z,K,random=False):
    print('calculating prior...')
    Po = np.ones(4**K)/4**K
    if random:
        return Po
    T = len(Z)
    for t in range(T):
      Po[Z[t]] = Po[Z[t]]+1
    row_sums = Po.sum()
    Po = Po/row_sums
    #print('the prior by MLE is:'+"\n"+str(Po))
    print('done')
    return Po
  # calculate transition table based on MLE
  def transition_init(Z_,K,random=False):
    print('caluating transition matrix by MLE...')
    T = len(Z_)
    A = HMM_DNA.A_init(K)
    if random:
      return A
    for t in range(T):
      Z = Z_[t]
      Za = HMM_DNA.base2num([Z[0]],K)
      Zb = HMM_DNA.base2num([Z[1]],K)
      #if Za == 4**K-1 and Zb<252:
      #  print(Z)
      #  print(t)
      A[Za,Zb] = A[Za,Zb]+1
    row_sums = A.sum(axis=1)
    A = A/row_sums[:,np.newaxis]/5
    A = A + np.diag(np.ones(4**K)*4/5)
    #print('the transition matrix by MLE is:'+"\n"+str(A))
    print('done')
    return A
  
  #########################################################################################################
  # compute the observation table based on pre-trained NN
  def obs_table(path,X,P_xz,Po,T,K,myfile,write_file):
    #print('calculating the neural network output...')
    bench_size = 5000
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name("x_input:0")
    y_ = graph.get_tensor_by_name("y_:0")
    pkeep = graph.get_tensor_by_name("pkeep:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    y = graph.get_tensor_by_name('y:0')
    class_nn = np.zeros(T) 
    if write_file:
      myfile.write("OUTPUT from NN: \n")
    for t in range(0,T,bench_size):
      X_ = X[t:t+bench_size,:]
      feed_dict={x_input:X_,y_:np.zeros((bench_size,K)),is_training:[0],pkeep:1.0}
      p_zx = sess.run(y,feed_dict)
      if write_file:
        myfile.write(str(p_zx)+"\n")
      current_size,_ = np.shape(p_zx)
      for i in range(current_size):
        class_nn[t+i] = np.argmax(p_zx[i,:]) #classification output from NN
        P_xz_ = p_zx[i,:]/Po
        row_sums = P_xz_.sum()
        P_xz[t+i,:] = P_xz_/row_sums # observation probability table
        if t+i < 0:
          print('nn output:'+str(p_zx[i,:]))
          print('nn output nor:'+str(P_xz_/row_sums))
    print('done')
    return [P_xz,class_nn]
  #def obs_table_update(Y,Po,T):
  #  for t in range(T):
  #    P_xz_ = Y[t,:]/Po
  #    row_sums = P_xz_.sum()
  #    P_xz[t,:] = P_xz_#/row_sums # observation probability table
  #  return P_xz
  #######################################################################################################
  # the update of transition matrix with BW
  def xi_update(alpha,beta,A,P_xz,Gamma,T,K):
    xi = np.zeros((T-1,K*K))
    for t in range(T-1):
      for k in range(K):
        xi[t,(k*K):((k+1)*K)] = alpha[t,k]*A[k,:]*P_xz[t+1,:]*beta[t+1,:]
      #print("xi(t): "+str(xi[t,:]))
      xi[t,:] = xi[t,:]/np.sum(xi[t,:])
    xi_sum = xi.sum(axis=0)
    #print("xi_sum: "+str(xi.sum(axis=1)))
    Gamma_sum = Gamma.sum(axis=0)
    #print("Gamma_sum: "+str(Gamma.sum(axis=1)))
    for k in range(K):
      A[k,:] = xi_sum[k*K:(k+1)*K]/Gamma_sum[k]
      #print("A: "+str(A[k,:]))
      A[k,:] = A[k,:]/np.sum(A[k,:])
    return [A,xi]
  #######################################################################################################
  # The Viterbi
  # implement the Viterbi algorithm
  def viterbi(Pi,P_xz,A,T,K):
    print('calculating the Viterbi algorithm...')
    print('the algorithm will evaluate a sequence of '+ str(T) +' with ' +str(K) + ' classes')
    Kxi = np.zeros((T,K))
    Kxi[0,:] = np.multiply(Pi,P_xz[1,:])
    print('Kxi: '+str(Kxi[0,:]))
    Ita = np.zeros((T-1,K))
    for t in range(1,T):
      Kxi_sum = 0
      for k in range(K):
        Kxi[t,k] = np.multiply(P_xz[t,k],np.max(np.multiply(Kxi[t-1,:],A[:,k])))
        Kxi_sum = Kxi_sum+Kxi[t,k]
        Ita[t-1,k] = np.argmax(np.multiply(Kxi[t-1,:],A[:,k]))
      Kxi[t,:] = Kxi[t,:]/Kxi_sum
    Z_hat = np.zeros(T)
    Z_hat[T-1] = np.argmax(Kxi[T-1,:])
    for t in reversed(range(T-1)):
      #print(t)
      Z_hat[t] = int(Ita[t,int(Z_hat[t+1])])
    print('done')
    return [Z_hat,Kxi,Ita]
  ##########################################################################################################
  # The Forward and Backward propogation
  def ab(Pi,P_xz,A,T,K):
    # forwards
    alpha = np.zeros((T,K))
    alpha_ = np.multiply(Pi,P_xz[0,:])
    alpha[0,:] = alpha_/alpha_.sum()
    for t in range(1,T):
      for k in range(K):
        alpha_[k] = P_xz[t,k]*np.sum(np.multiply(alpha[t-1,:],A[:,k]))
      alpha[t,:] = alpha_/alpha_.sum()
    # backward
    beta = np.zeros((T,K))
    beta_ = np.ones(K)
    beta[T-1,:] = beta_
    for t in range(T-2,-1,-1):
      for k in range(K):
        beta_[k] = np.sum(np.multiply(np.multiply(A[:,k],P_xz[t+1,:]),beta[t+1,:]))
      beta[t,:] = beta_/np.sum(beta_)
    #estimate the posterior for z_t
    gamma = np.zeros((T,K))
    z_t = np.zeros(T)
    for tt in range(T):
      gamma[tt,:] = np.multiply(alpha[tt,:],beta[tt,:])
      gamma[tt,:] = gamma[tt,:]/np.sum(gamma[tt,:])
      z_t[tt] = np.argmax(gamma[tt,:])
    return [z_t,gamma,alpha,beta]
  #########################################################################################################
  # the mismatch rate of the most possible states sequence
  def acc(A,B,T):
    eq=0
    for t in range(T):
      if np.isclose(A[t],B[t]):
        eq = eq+1
    eq = np.true_divide(eq,T)
    return eq
  ##########################################################################################################
  @staticmethod
  def assembler(Z_idx,Z_base,K,repeat_len = 12):
    repeat_idx = np.zeros(4)
    for i in range(4):
      for j in range(K):
        repeat_idx[i] += i*(4**j)
    repeat_test = HMM_DNA.base2num(repeat_idx,K,op=True)
    #print(repeat_test)
    T = len(Z_idx)
    base = ''
    for i in range(T):
      #isspecial = len(np.where(repeat_test==Z_idx[i])[0])>0
      if len(base)==0:
        base = base + Z_base[i]
      elif Z_idx[i-1]==Z_idx[i] and len(np.where(repeat_test==Z_idx[i])[0])>0:
        repeat = repeat + 1
        if repeat > repeat_len:
          base = base + Z_base[i][-1]
          repeat = 0
        continue
      elif Z_idx[i-1]!=Z_idx[i]:
        repeat = 0
        base = base + Z_base[i][-1]
    return base
  def manutal_math(base_a,base_b,filename='true&us'):
    seg_size = 12
    #########################################################
    search_r = len(base_a)-seg_size 
    neighb = 1000
    ali_idx_l = np.zeros(search_r)
    ali_idx_m = np.zeros(search_r)
    ali_idx_h = np.zeros(search_r)
    print("finding the matches between chiron base and our base...")
    for i in tqdm.trange(search_r):
      segment = base_a[i:i+seg_size]
      for j in range(len(base_b)-seg_size):#range(max(0,i-neighb),min(len(base_b)-seg_size,i+neighb)):
        count = 0
        for k in range(seg_size):
          if(segment[k] == base_b[j+k]):
            count = count + 1
        count = count/seg_size
        if(count==1):
          ali_idx_h[i] = j
        elif(count > 0.91):
          ali_idx_m[i] = j
        #elif(count>0.83):
          #ali_idx_l[i] = j
        #if(segment == base_b[j:j+seg_size]):
        #  ali_idx[i] = j
        #  break
    print("find full match : "+str(np.count_nonzero(ali_idx_h))+' find middle match : '+str(np.count_nonzero(ali_idx_m))+' find low match(default to ignore) '+str(np.count_nonzero(ali_idx_l)))
    nzl = np.nonzero(ali_idx_l)[0] #1113 / 1479
    nzm = np.nonzero(ali_idx_m)[0] #1197 / 1687
    nzh = np.nonzero(ali_idx_h)[0] # 0
    plt.figure(1)
    plt.plot(nzl,ali_idx_l[nzl],'g+',nzm,ali_idx_m[nzm],'b+',nzh,ali_idx_h[nzh],'r+')
    fname = 'base_compair'+filename
    plt.savefig(fname,format='png')
    plt.close(plt.figure(1))

  def __init__(self,export_op=True,max_iter=1,using_viterbi=0,using_ab=0,Kmer=3,write_file=0,test_another=0):
    # set parameters
    #print(base2num(['AAA','AAA','AAC','AAG','AAT'],3))
    #print(base2num(['AAC','ACA','ACC','ACG','ACT'],3))
    #print(base2num(['AAG','AGA','AGC','AGG','AGT'],3))
    #print(base2num(['ACA','CAA','CAC','CAG','CAT'],3))
    
    #print(base2num([1,2],3,op= True))
    if export_op:
      #base_test = ['TGC','GCT','CTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TTT','TAC','ACT','ACT','ACT','ACT','ACT','ACT','ACT','ACT','ACT','CTG','TGA','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','GAC','ACG']
      #base_test_idx = HMM_DNA.base2num(base_test,3)
      
      #self.base_con = HMM_DNA.assembler(Z_idx = base_test_idx,Z_base=base_test,K=3)
      return None
    K = 4**Kmer
    if write_file:
      myfile = open('HMM_BW.txt','w')
    #############################################
    # read DNA data
    print('reading in the DNA dataset')
    dataset = DNAData(k=Kmer)
    Z_train, X_test, Z_test, Z_o = dataset.get_ZX()
    if(test_another):
      X_test2, Z_test2 = dataset.get_another_ZX()
    #############################################
    # initialization
    print("initializing...")
    Z_o = HMM_DNA.base2num(Z_o,Kmer)
    Po = HMM_DNA.prior_init(Z_o,Kmer) #prior
    A = HMM_DNA.transition_init(Z_train,Kmer,random=False) #transition
    Pi = Po # initial states
    if write_file:
      myfile.write('Initial state Po: \n'+str(Po)+'\n')
      myfile.write('Transition matrix A: \n'+str(A)+'\n')
      
    #############################################
    #esetimated observation
    model_path = 'fivedatamodel-400'
    for X_ in X_test:
      T,_ = np.shape(X_)
      #tempf = TemporaryFile()
      f = tables.open_file('P-xz.h5','w')
      root = f.root
      P_xz = f.create_carray(root,'P_xz',tables.Float64Atom(),shape=(T,K))
      P_xz,class_nn= HMM_DNA.obs_table(model_path,X_,P_xz,Po,T,K,myfile,write_file) 
      #############################################
      for i in range(max_iter):
        if using_viterbi:
          Z_hat,Kxi,Ita = HMM_DNA.viterbi(Pi,P_xz,A,T,K)
          #myfile.write('Kxi: \n'+str(Kxi)+'\n')
          #myfile.write('Ita: \n'+str(Ita)+'\n')
          myfile.write('Z_hat; \n'+str(Z_hat)+'\n')
          #accuracy = acc(Z_hat,Z,T)
          #print('Viterbi: '+str(accuracy))
          Z_base = HMM_DNA.base2num(Z_hat,Kmer,op=True)
          NN_base = HMM_DNA.base2num(class_nn,Kmer,op=True)
          myfile.write('Z_base from Viterbi:'+str(Z_base)+'\n')
          myfile.write('Z_base from Classification:'+str(NN_base)+'\n')
          Z_base_ass = HMM_DNA.assembler(Z_hat,Z_base,Kmer)
          NN_base_ass = HMM_DNA.assembler(class_nn,NN_base,Kmer)
          myfile.write('true result:\n'+str(Z_test[i])+'\n')
          myfile.write('final base from Viterbi:\n'+str(Z_base_ass)+'\n')
          myfile.write('final base from Classification:\n'+str(NN_base_ass)+'\n')
          HMM_DNA.manutal_math(Z_test[i],Z_base_ass,'us&chiron7')
          HMM_DNA.manutal_math(Z_test[i],NN_base_ass,'NN&chiron7')
        if using_ab: # Baum-Welch update process/viterbi decode
          Z_hat,Gamma,alpha,beta = HMM_DNA.ab(Pi,P_xz,A,T,K)
          Pi = Gamma[1,:] # update Prior Po
          #P_xz = obs_table_update(Y,Po,T) # update the observation probability table
          A,xi = HMM_DNA.xi_update(alpha,beta,A,P_xz,Gamma,T,K) # update the transition probability
          #myfile.write('Gamma: \n'+str(Gamma)+'\n')
          #myfile.write('alpha: \n'+str(alpha)+'\n')
          #myfile.write('beta: \n'+str(beta)+'\n')
          #myfile.write('xi: \n'+str(xi)+'\n')
          myfile.write('A: \n'+str(A)+'\n')
          accuracy = HMM_DNA.acc(Z_hat,Z,T)
          print('Forward-Backward:'+str(accuracy))
      #myfile.write('TRUE HIDDEN STATES: \n'+str(Z)+'\n')
      #myfile.write('ESTIMATE HIDDEN STATES: \n'+str(Z_hat)+'\n')
      f.close()
      myfile.close()
      accuracy = acc(Z_hat,Z,T)
      
      zero_hat = np.zeros(T)
      acc_zero = acc(zero_hat,Z,T)
      acc_nn = acc(class_nn,Z,T)
      print('the classification rate with NN is: '+str(acc_nn))
      if(trainrat1==arat):
        print('the classification rate with NN is overestimated!')
      print(' the accuracy of predicting 0 all the time is ' + str(acc_zero))
