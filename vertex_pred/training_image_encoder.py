#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:23:16 2018

@author: aidanrocke
"""

import random
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob

from image_encoder import image_encoder

##  1. define data file paths:

# training_data = 

order_key = lambda p: int(p.rsplit("/", 1)[1].split(".")[0])

## 2. load files:
images = sorted(glob.glob(training_data+'imgs/*.npy'),key=order_key)
edges = sorted(glob.glob(training_data+'edges/*.npy'),key=order_key)
vertices = sorted(glob.glob(training_data+'polys/*.npy'),key=order_key)

## define mapping from files to numpy arrays:
def files_to_array(file_list,ix):
    
    A = []
    for i in ix:
        array = np.load(file_list[i])
        A.append(array)
    
    return np.stack(A)

def training_batch(batch_size):

    N, n = len(images), 500
    
    ix_1 = random.choice(np.arange(N))
    ix_2 = (ix_1 + n)*(ix_1 + n < N) +  (ix_1 - n)*(ix_1 + n >= N)
        
    a, b = min([ix_1,ix_2]), max([ix_1,ix_2])

    ix = np.random.choice(n,batch_size,replace=False)

    X = files_to_array(images[a:b],ix)
    Y = files_to_array(edges[a:b],ix)
    Z = files_to_array(vertices[a:b],ix)

    return X,np.expand_dims(Y,axis=-1),np.expand_dims(Z,axis=-1)

## 3. define models:
random_seed = 42

tf.reset_default_graph()

vgg_model = image_encoder(random_seed)

## 4. define method for saving models:

# export_dir = 

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

## 5. run training:
batch_size = 16
batch_iterations = round(len(images)/batch_size)
epochs = 110

## visual evaluation path:
# eval_path = 

# save loss to text file:
f = open(eval_path+"loss.txt","w+")

with tf.Session() as sess:
    
    sess.run(vgg_model.init_g)
    
    ## train:
    for i in range(epochs):
        
        ## decay the learning rate by a factor of 10 every 10 epochs:
        learning_rate = 0.0001/(10**(i // 10))
        
        for j in range(batch_iterations):
        
            train_images, train_edges,train_vertices = training_batch(batch_size) 
            ## update parameters:
            
            train_feed = {vgg_model.input_image:train_images,
                          vgg_model.edges:train_edges,
                          vgg_model.vertices: train_vertices,
                          vgg_model.prob:0.9,
                          vgg_model.lr: learning_rate}
            
            sess.run(vgg_model.optimizer,feed_dict = train_feed)
                
        if i % 20 == 0:

            ## do a visual evaluation:
            eval_x, eval_y, eval_z = training_batch(1)

            eval_feed = {vgg_model.input_image:eval_x}

            v_hat, e_hat = sess.run([vgg_model.vertices_hat],feed_dict = eval_feed), sess.run([vgg_model.edges_hat],feed_dict = eval_feed)

            v_hat, e_hat = np.array(v_hat), np.array(e_hat) ## convert list to array

            fig, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(np.reshape(eval_y,(28,28)))  ## ground truth edges
            axarr[0,1].imshow(e_hat[0,0,:,:,0])
            axarr[1,0].imshow(np.reshape(eval_z,(28,28)))
            axarr[1,1].imshow(v_hat[0,0,:,:,0]) ## ground truth vertices

            plt.savefig(eval_path+'visual_'+str(i)+'.png')


            eval_images, eval_edges,eval_vertices = training_batch(32) 

            eval_feed = {vgg_model.input_image:eval_images,
                          vgg_model.edges:eval_edges,
                          vgg_model.vertices: eval_vertices}

                    
            ## check the loss:
            loss_value = sess.run([vgg_model.loss], eval_feed)
            print('loss', loss_value)

            ## write loss to text file:
            f.write("The loss is %.2f\n" % loss_value[0])

            saver.save(sess,export_dir,global_step=i)


f.close() 
