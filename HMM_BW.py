from __future__ import division
import time
import numpy as np
import tensorflow as tf
from sleep_read import SleepScore

np.set_printoptions(threshold=np.nan)
#CHECKPOINT_PATH = ./checkpoints/
###########################################################################################################
# calculate transition table based on MLE
def transition_init(Z1,Z2,K,random=False):
  T1 = np.size(Z1)
  T2 = np.size(Z2)
  A = np.random.rand(K,K)*0.01
  if random:
    row_sums = A.sum(axis=1)
    A = A/row_sums[:,np.newaxis]
    return A
  for t in range(T1-1):
    A[Z1[t],Z1[t+1]] = A[Z1[t],Z1[t+1]]+1
  for t in range(T2-1):
    A[Z2[t],Z2[t+1]] = A[Z2[t],Z2[t+1]]+1
  row_sums = A.sum(axis=1)
  A = A/row_sums[:,np.newaxis]
  print('the transition matrix by MLE is:'+"\n"+str(A))
  return A
# calculate prior of X based on MLE
def prior_init(Z1,Z2,K,random=False):
  T1 = np.size(Z1)
  T2 = np.size(Z2)
  Po = np.ones(K)/K
  if random:
    return Po
  for t in range(T1):
    Po[Z1[t]] = Po[Z1[t]]+1
  for t in range(T2):
    Po[Z2[t]] = Po[Z2[t]]+1
  row_sums = Po.sum()
  Po = Po/row_sums
  print('the prior by MLE is:'+"\n"+str(Po))
  return Po
#########################################################################################################
# compute the observation table based on pre-trained NN
def obs_table_init(path,X,Po,T,K):
  sess = tf.Session()
  saver = tf.train.import_meta_graph(path+'.meta')
  saver.restore(sess,tf.train.latest_checkpoint('./'))
  graph = tf.get_default_graph()
  x_input = graph.get_tensor_by_name("x_input:0")
  y_ = graph.get_tensor_by_name("y_:0")
  pkeep = graph.get_tensor_by_name("pkeep:0")
  is_training = graph.get_tensor_by_name("is_training:0")
  y = graph.get_tensor_by_name('y:0')
  P_xz = np.zeros((T,K))
  class_nn = np.zeros(T) 

  feed_dict={x_input:X,y_:np.zeros((T,K)),is_training:[0],pkeep:1.0}
  p_zx = sess.run(y,feed_dict)
  for t in range(T):
    class_nn[t] = np.argmax(p_zx[t,:]) #classification output from NN
    P_xz_ = p_zx[t,:]/Po
    row_sums = P_xz_.sum()
    P_xz[t,:] = P_xz_/row_sums # observation probability table
  return [P_xz,class_nn,p_zx]
def obs_table_update(Y,Po,T):
  for t in range(T):
    P_xz_ = Y[t,:]/Po
    row_sums = P_xz_.sum()
    P_xz[t,:] = P_xz_#/row_sums # observation probability table
  return P_xz
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
  Kxi = np.zeros((T,K))
  Kxi[0,:] = np.multiply(Pi,P_xz[1,:])
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
# set parameters
max_iter = 1
using_viterbi = 1
using_ab = 1
trainrat1 = 3
trainrat2 = 0
arat=3
write_file = 1 # write some results

if write_file:
  myfile = open('sleepscore/HMM_BW.txt','w')
#############################################
# read SleepScore data
print('reading in the SleepScoring dataset')
train_data1 = SleepScore(trainrat1)
Z_t1,X_t1 = train_data1.get_ZX()
if(trainrat2):
  train_data2 = SleepScore(trainrat2)
  Z_t2,X_t2 = train_data2.get_ZX()
else:
  Z_t2 = []
dataset = SleepScore(arat)
Z,X = dataset.get_ZX()
T = np.size(Z)
K = np.max(Z)+1
#############################################
# initialization
print("initializing...")
Po = prior_init(Z_t1,Z_t2,K,random=False) #prior
A = transition_init(Z_t1,Z_t2,K,random=False) #transition
Pi = Po # initial states
#############################################
#esetimated observation
model_path = 'class_model-2000'
P_xz,class_nn,Y = obs_table_init(model_path,X,Po,T,K) 

#############################################
for i in range(max_iter):
  if using_viterbi:
    Z_hat,Kxi,Ita = viterbi(Pi,P_xz,A,T,K)
    myfile.write('Kxi: \n'+str(Kxi)+'\n')
    myfile.write('Ita: \n'+str(Ita)+'\n')
    accuracy = acc(Z_hat,Z,T)
    print('Viterbi: '+str(accuracy))
  if using_ab: # Baum-Welch update process/viterbi decode
    Z_hat,Gamma,alpha,beta = ab(Pi,P_xz,A,T,K)
    Pi = Gamma[1,:] # update Prior Po
    #P_xz = obs_table_update(Y,Po,T) # update the observation probability table
    A,xi = xi_update(alpha,beta,A,P_xz,Gamma,T,K) # update the transition probability
    #myfile.write('Gamma: \n'+str(Gamma)+'\n')
    #myfile.write('alpha: \n'+str(alpha)+'\n')
    #myfile.write('beta: \n'+str(beta)+'\n')
    #myfile.write('xi: \n'+str(xi)+'\n')
    myfile.write('A: \n'+str(A)+'\n')
    accuracy = acc(Z_hat,Z,T)
    print('Forward-Backward:'+str(accuracy))
#myfile.write('TRUE HIDDEN STATES: \n'+str(Z)+'\n')
#myfile.write('ESTIMATE HIDDEN STATES: \n'+str(Z_hat)+'\n')
myfile.close()
accuracy = acc(Z_hat,Z,T)

zero_hat = np.zeros(T)
acc_zero = acc(zero_hat,Z,T)
acc_nn = acc(class_nn,Z,T)
print('the classification rate with NN is: '+str(acc_nn))
if(trainrat1==arat):
  print('the classification rate with NN is overestimated!')
print(' the accuracy of predicting 0 all the time is ' + str(acc_zero))
