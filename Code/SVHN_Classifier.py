#Import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sc
import scipy.misc
import numpy as np
import numpy.random as rand
import argparse
import sys
import math


#Load Data

width = 32
height = 32
channels = 3

def load_mat_dataset(mat_dataset):
    images1=mat_dataset['X']
    images=[]
    #print(np.shape(images1))
    for i in range(np.shape(images1)[3]):
        images.append(images1[:width,:height,:channels,i])
    
    labels1=mat_dataset['y']
    labels=[]
    for i in labels1:
        a=[]
        for j in range(10):
            if(i==j or (j==0 and i==10)):
                a.append(1)
            else:
                a.append(0)
        labels.append(a)
        
    return images, labels

def create_svhn_batch(images, labels, batch_size, step):
    # return batch as numpy array = [num_images, w, h, c]
    # return labels
    j=0
    batch=[[],[]]
    random=rand.rand(batch_size)*np.shape(images)[0]
    #print(np.shape(random))
    
    for i in range(batch_size-1):
        #print(i)
        batch[0].append(images[int(random[i])])
        batch[1].append(labels[int(random[i])])
    return batch


train_raw = sc.loadmat('/home/pfe-mgrbic/storage/train_32x32.mat')
test_raw = sc.loadmat('/home/pfe-mgrbic/storage/test_32x32.mat')

train_data=load_mat_dataset(train_raw)
test_data=load_mat_dataset(test_raw)

'''
for j in range(10):
    for i in range(20000):
        image=scipy.misc.imread("/home/pfe-mgrbic/storage/SyntSynt/Synt/line-"+str(j)+"-"+str(i)+".png")
        a=[]
        for l in range(10):
            if(l==j):
                a.append(1)
            else:
                a.append(0);
        train_data[0].append(image)
        train_data[1].append(a)
 '''       


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2dstr2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def deepnn(x):
    
    #Convolution1
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    #Convolution2
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    
    #Pooling1
    #h_pool1 = max_pool_2x2(h_conv2)

    #Convolution3
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    
    #Convolution4
    W_conv4 = weight_variable([5, 5, 128, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    #Pooling2
    #h_pool2 = max_pool_2x2(h_conv4)
    
    #Convolution5
    W_conv5 = weight_variable([5, 5, 128, 256])
    b_conv5 = bias_variable([256])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    
    #Convolution6
    W_conv6 = weight_variable([5, 5, 256, 256])
    b_conv6 = bias_variable([256])
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    #Pooling3
    #h_pool3 = max_pool_2x2(h_conv6)
    
    

    #Dense layer1
    W_fc1 = weight_variable([32 * 32 * 256, 128])
    b_fc1 = bias_variable([128])
    
    h_flat1 = tf.reshape(h_conv6, [-1, 32*32*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat1, W_fc1) + b_fc1)
    
    #Dropout1
    keep_prob = tf.placeholder(tf.float32)
    h_drop1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #Dense layer2
    W_fc2 = weight_variable([128,128])
    b_fc2 = bias_variable([128])
    h_fc2 = tf.nn.relu(tf.matmul(h_drop1, W_fc2) + b_fc2)
    
    #Dropout2
    h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)
  
    #Last layer
    W_fc3 = weight_variable([128, 10])
    b_fc3 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3
    return y_conv, keep_prob



x = tf.placeholder(tf.float32, [None, 32,32,3])
y_ = tf.placeholder(tf.float32, [None, 10])
y_conv, keep_prob = deepnn(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()   

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(80001):        
        
        batch = create_svhn_batch(train_data[0],train_data[1],50,i)
        
        
        #train_accuracy = accuracy.eval(feed_dict={
        #x: batch[0], y_: batch[1], keep_prob: 1.0})
        
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        if(i%100==0):
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.})
            print("step "+ "{}".format(i) +", training accuracy "+"{}".format(acc)+", loss " + "{:.6f}".format(loss))
         

        if(i%1000==0):
            validation_acc=0.0

            for j in range(int(len(test_data[1])/100)):
                batch = create_svhn_batch(test_data[0],test_data[1],100,j)
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.})
                validation_acc+=float(acc)
                
            validation_acc/= len(test_data[1])/100   
            print("step "+ "{}".format(i) +", validation accuracy "+"{}".format(validation_acc))
        
    
    
    save_path = saver.save(sess, "save/modelSVHN.ckpt")
    
