# simple python script to train a density NN 
import os
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
from creat_data import DNAData
import matplotlib.pyplot as plt
from HMM_BW_DNA import HMM_DNA
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def bnorm(x,op='normal',mu = 75, sigma = 25):
  if op == 'fixed':
    x = (x-75)/25
    return x
  if op == 'no':
    return x
  if op == 'normal':
    shape = x.get_shape().as_list()
    
    gamma = tf.Variable(tf.ones(shape[len(shape)-1]))
    beta = tf.Variable(tf.zeros(shape[len(shape)-1]))
    batch_mean,batch_var = tf.nn.moments(x,np.arange(len(shape)).tolist())
    with tf.control_dependencies([batch_mean,batch_var]):
      #u_mean = batch_mean*is_training[0]+pop_mean*(1-is_training[0])
      #u_var = batch_var*is_training[0] + pop_var*(1-is_training[0])
      next_x = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,gamma,10e-08)
  return next_x
############################################################
write_file = 1
write_norm = 0
write_pred = 1
using_tensorboard = 0
training = 1
export_model = 1
############################################################
# tuning parameters
learning_rate=0.001
beta1=0.9
beta2=0.999
epsilon=1e-08
use_locking=False
name='Adam'
nbatch = 5000
n_iter = 500
eval_steps = 50
dropout = 0.25
kmer = 3
another = 1 # if test on multiple sets
###########################################################
# creating and reading the DNA dataset
print('creating the DNA dataset')

dataset = DNAData(k=kmer,write_file=False)
#dataset.check_balance()
input_train,output_train,input_test, output_test= dataset.get_data()
if another:
  input_test2, output_test2 = dataset.get_another_data()
print('train set size: '+str(np.shape(input_train)[0]))
print('test set size: '+str(np.shape(input_test)[0]))
print('data finished')
############################################################
# density network output 3 probabilities
input_dim = 300
output_dim = 4**kmer
x_input = tf.placeholder(tf.float32, shape = [None, input_dim],name = 'x_input')
y_ = tf.placeholder(tf.float32, shape = [None,output_dim],name='y_')
pkeep = tf.placeholder(tf.float32,name='pkeep')
is_training = tf.placeholder(tf.float32,shape = 1,name='is_training')
# weights and bias
M1 = 500
M2 = 100
#output_list = []
#M3 = 200
M4 = output_dim

W1 = tf.Variable(tf.truncated_normal([input_dim, M1], stddev=.01))
b1 = tf.Variable(tf.constant(0.1, shape=[M1]))
W2 = tf.Variable(tf.truncated_normal([M1, M2], stddev=.01))
b2 = tf.Variable(tf.constant(0.1, shape=[M2]))
#W3 = tf.Variable(tf.truncated_normal([M2, M3], stddev=.01))
#b3 = tf.Variable(tf.constant(0.1, shape=[M3]))
W4 = tf.Variable(tf.truncated_normal([M2,M4], stddev=.01))
b4 = tf.Variable(tf.constant(0.1, shape=[M4]))

# model with batch normalization
x_norm = bnorm(x_input,op='fixed')
Hf1 = tf.matmul(x_norm ,W1) + b1
H1 = tf.nn.relu(Hf1)
Hf2 = tf.matmul(H1,W2) + b2
H2 = tf.nn.relu(Hf2)
#Hf3 = tf.matmul(H2,W3) + b3
#H3 = tf.nn.relu(Hf3,is_training)
H4 = tf.matmul(H2,W4) + b4
y = tf.nn.softmax(H4,name='y')
#y_shape = H4.get_shape()
#for i in range(kmer):
  #output_list.append(tf.nn.softmax(H4[:,i*4:(i+1)*4]))
#y = tf.concat(output_list,axis=1, name='y')
#print(y.get_shape)
# lost function
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

global_step = tf.Variable(0,dtype = tf.int32,trainable=False,name='global_step')
train_step = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon,use_locking,name).minimize(cross_entropy,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

init = tf.global_variables_initializer()
###########################################################################################
if write_file:
  myfile = open('output.txt','w')
  tvars = tf.trainable_variables()
if using_tensorboard:
  # keep track of the loss and accuracy for the training set
  tf.summary.scalar('training loss', cross_entropy, collections=['training'])
  tf.summary.scalar('training accuracy', accuracy, collections=['training'])
  # merge the two quantities
  tsummary = tf.summary.merge_all('training')

  # keep track of the loss and accuracy for the validation set
  tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
  tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
  # merge the two quantities
  vsummary = tf.summary.merge_all('validation')

###############################################################################################
time_start = time.clock()


# 2.1) start a tensorflow session
if training:
  with tf.Session() as sess:
  ##################################################i
    if using_tensorboard:
  # set up a file writer and directory to where it should write info + 
  # attach the assembled graph
      summary_writer = tf.summary.FileWriter('tensorboard/results', sess.graph)
    if export_model:
      saver = tf.train.Saver()
    ##################################################
    # 2.2)  Initialize the network's parameter variables
    # Run the "init" op (do this when training from a random initialization)
    sess.run(init)
    epo=0
    indx = np.arange(0,np.size(input_train,0))
    # 2.3) loop for the mini-batch training of the network's parameters
    for i in range(n_iter):
        if (i+1)*nbatch>(epo+1)*np.size(input_train,0):
          epo=epo+1
          np.random.shuffle(indx)
        train_input=input_train[indx[max(0,(i)*nbatch-epo*np.size(input_train,0)):(i+1)*nbatch-epo*np.size(input_train,0)],:]
        train_output=output_train[indx[max(0,(i)*nbatch-epo*np.size(input_train,0)):(i+1)*nbatch-epo*np.size(input_train,0)]]
        batch_dict = {
            x_input: train_input, # input data
            y_: train_output, # corresponding label
            is_training:[1.0],
            pkeep:dropout
         }
        #####################################################################
        # output and plot the normalization
        #print("mean: "+str(b_mean_)+" var: "+str(b_var_))
        if write_norm:
          x_norm_ = sess.run(x_norm,feed_dict = {x_input:train_input,y_:train_output,is_training:[0.0],pkeep:[1]})
          myfile.write('original input:'+str(train_input)+'\n')
          #myfile.write('the normalization mean:'+ str(b_mean_)+' and var: '+str(b_var_))
          myfile.write('normalized input:'+str(x_norm_)+'\n')
          plt.figure(i)
          plt.subplot(2,1,1)
          batch_n,_ = np.shape(train_input)
          for batch_i in range(5):
            plt.plot(np.arange(batch_i*300,(batch_i+1)*300),train_input[batch_i,:])
          plt.subplot(2,1,2)
          for batch_i in range(5):
            plt.plot(np.arange(batch_i*300,(batch_i+1)*300),x_norm_[batch_i,:])
          fname = 'output'+str(i)
          plt.savefig(fname,format='png')
          plt.close(plt.figure(i))
        ######################################################################
       
        sess.run(train_step, feed_dict=batch_dict)
        
        # periodically evaluate how well training is going
        if i % eval_steps == 0:
            #save_path = saver.save(sess,'./checkpoints/pre_model',global_step = i)
            #savefile.write(save_path + '\n')

            # compute the performance measures on the training set by
            # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"
            if using_tensorboard:
              tc,ta,tt = sess.run([cross_entropy, accuracy,tsummary], feed_dict = {x_input:train_input, y_: train_output,is_training:[1],pkeep: dropout})
            else:
              tc,ta = sess.run([cross_entropy, accuracy], feed_dict = {x_input:input_train, y_: output_train,is_training:[1],pkeep: dropout})
            # compute the performance measures on the validation set by
            # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"
            if using_tensorboard:
              pred,vc,va,vv = sess.run([y,cross_entropy, accuracy,vsummary], feed_dict={x_input:input_test, y_:output_test,is_training : [0],pkeep:1.0})            
              info = str(i)+" train acc " + str(ta) +' test acc '+ str(va) 
              if another!=0:
                w1,pred2,vc2,va2,vv2 = sess.run([W1,y,cross_entropy, accuracy,vsummary], feed_dict={x_input:input_test2, y_:output_test2,is_training : [0],pkeep:1.0})
                info = str(i)+" train acc " + str(ta) +' test acc '+ str(va) + ' test on another fold '+str(va2)
            else:
              pred,vc,va = sess.run([y,cross_entropy, accuracy], feed_dict={x_input:input_test, y_:output_test,is_training : [0],pkeep:1.0})            
              info = str(i)+" train acc " + str(ta) +' test acc '+ str(va) 
              if another!=0:
                w1,pred2,vc2,va2 = sess.run([W1,y,cross_entropy, accuracy], feed_dict={x_input:input_test2, y_:output_test2,is_training : [0],pkeep:1.0})
                info = str(i)+" train acc " + str(ta) +' test acc '+ str(va) + ' test on another fold '+str(va2)
            
            print(info)
            if using_tensorboard:
              summary_writer.add_summary(tt, i)
              summary_writer.add_summary(vv, i)
            if write_pred:
              myfile.write('step:'+str(i)+'\n'+str(pred2)+'\n')
              myfile.write('\n')
    #savefile.close()
    # evaluate the accuracy of the final model on the test data
    if write_file:
      tvars_vals,test_acc = sess.run([tvars,accuracy], feed_dict={x_input: input_test2, y_:output_test2,is_training : [0],pkeep:1.0})
    else:
      test_acc = sess.run(accuracy, feed_dict={x_input: input_test2, y_:output_test2,is_training : [0],pkeep:1.0})
    #print(tvars)
    time_cost = time.clock()-time_start
    final_msg = "with learning rate:"+str(learning_rate)+", batch size:"+str(nbatch)+", iters:"+str(n_iter)+', the test accuracy is:' + str(test_acc)+"\ncomputional time cose:"+str(time_cost)+"s"
    print(final_msg)
    if export_model:
      save_path = saver.save(sess,'models/fivedatamodel',global_step = n_iter)
      print("Model saved in path: %s" %save_path)
    #print('W1:%s' %W1.eval())
if write_file:
  #for var, var in zip(tvars, tvars_vals):
    #if var.name == "W1":
      #myfile.write(str(var.name)+'\n'+str(val)+'\n')
  if write_pred:  
    myfile.write('predicted states:'+str(pred)+'\n')
    myfile.write('predicted states:'+str(HMM_DNA.base2num(np.argmax(pred2,axis=1),kmer,op=True))+'\n')
    myfile.write('true states S_t:'+str(HMM_DNA.base2num(np.argmax(output_test2,axis=1),kmer,op=True))+'\n')
  #myfile.write('input states S_(t-1):'+str(np.argmax(input_test[:,1000:1003],axis=1)+1)+'\n')
  #myfile.write('first layer weight on input is: '+'\n'+str(tf.get_collection(tf.GraphKeys.WIGHTS))+'\n')
  myfile.close()
