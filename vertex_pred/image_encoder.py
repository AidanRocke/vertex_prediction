#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 22:11:44 2018

@author: aidanrocke
"""
import numpy as np
import tensorflow as tf
from group_norm import GroupNormalization

## load the VGG-16 model:
VGG16 = tf.keras.applications.VGG16()

class image_encoder:
    
    def __init__(self,random_seed):
        self.seed = random_seed
        
        self.input_image = tf.placeholder(tf.float32, shape = [None, 224,224,3],name='X')
        self.prob = tf.placeholder_with_default(1.0, shape=(),name='prob')
        
        ## vertex prediction placeholders:
        self.lr = tf.placeholder(tf.float32, shape = [],name='lr')
        self.edges = tf.placeholder(tf.float32, shape = [None, 28,28,1],name='edges')
        self.vertices = tf.placeholder(tf.float32, shape = [None, 28,28,1],name='vertices')
        
        ## get VGG-16 weights:
        self.VGG16_weights = VGG16.get_weights()
        
        ## VGG-16 blocks:
        self.block_1 = self.block_1()
        self.block_2 = self.block_2()
        self.block_3 = self.block_3()
        self.block_4 = self.block_4()
        self.block_5 = self.block_5()
                
        ## concatenated outputs:
        self.fused_block = self.concatenate_blocks()
        
        ## image representation:
        self.representation = self.image_representation()

        self.out_1, self.out_2, self.out_3 = self.edges_and_vertices()
        self.vertex = tf.identity(self.out_3,name='first_vertex')
        self.edges_hat, self.vertices_hat = tf.identity(self.out_1,name='edges_hat'),tf.identity(self.out_2,name='vertices_hat') 
        self.image_rep = tf.identity(self.representation,name='image_rep')
        
        ## define loss function:
        self.loss = 0.5*tf.losses.log_loss(self.edges,self.edges_hat)+\
                    0.5*tf.losses.log_loss(self.vertices,self.vertices_hat) 

        ## define optimizer:
        self.optimizer = tf.train.AdamOptimizer(self.lr,name='adam').minimize(self.loss)
        
        ## initialise variables:
        self.init_g = tf.global_variables_initializer()  
        
    def conv2d(self,input_tensor,weight_init,bias_init,name,filter_shape=None):
    
        if filter_shape == None:
        
            kernel = tf.Variable(initial_value=weight_init)
            bias = tf.Variable(initial_value=bias_init)
            
        elif weight_init == 'glorot':
            
            H,W,c_in,c_out = filter_shape
    
            weight_init = np.random.rand(H,W,c_in,c_out).astype(np.float32) * \
                          np.sqrt(6.0/(c_in+c_out))
            bias_init =   np.zeros([c_out]).astype(np.float32)
            
            kernel = tf.Variable(initial_value=weight_init)
            bias = tf.Variable(initial_value=bias_init)
            
        
        conv = tf.nn.conv2d(input=input_tensor,filter=kernel,padding="SAME",
                            strides=(1,1,1,1),name=name)
        
        return tf.nn.bias_add(conv, bias)
    
    def block_1(self):
        """
            First block of VGG-16. 
            
            input: RGB image of dimension 224x224x3
            output: feature map of dimension 112x112x128
        """
        
        with tf.variable_scope("block_1",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            ## first convolution:
            conv_1 = self.conv2d(self.input_image,self.VGG16_weights[0],
                            self.VGG16_weights[1],"conv_1")
            
            relu_1 = tf.nn.relu(conv_1)
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_1)
            
            drop_1 = tf.nn.dropout(gnorm_1, self.prob)
            
            ## second convolution:
            conv_2 = self.conv2d(drop_1,self.VGG16_weights[2],
                            self.VGG16_weights[3],"conv_2")
            
            relu_2 = tf.nn.relu(conv_2)
            
            gnorm_2 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_2)
            
            drop_2 = tf.nn.dropout(gnorm_2, self.prob)
            
            # max pooling:            
            pool_1 = tf.nn.max_pool(drop_2,ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],padding='SAME')
        
        return pool_1
    
    def block_2(self):
        """
            Second block of VGG-16. 
            
            input: feature map of dimension 112x112x128
            output: feature map of dimension 56x56x128
        """
        
        with tf.variable_scope("block_2",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
                    
            ## first convolution:
            conv_1 = self.conv2d(self.block_1,self.VGG16_weights[4],
                            self.VGG16_weights[5],"conv_1")
            
            relu_1 = tf.nn.relu(conv_1)
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_1)
            
            drop_1 = tf.nn.dropout(gnorm_1, self.prob)
            
            ## second convolution:
            conv_2 = self.conv2d(drop_1,self.VGG16_weights[6],
                            self.VGG16_weights[7],"conv_2")
            
            relu_2 = tf.nn.relu(conv_2)
            
            gnorm_2 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_2)
            
            drop_2 = tf.nn.dropout(gnorm_2, self.prob)
            
            # max pooling:            
            pool_1 = tf.nn.max_pool(drop_2,ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],padding='SAME')
        
        return pool_1
    
    def block_3(self):
        """
            Third block of VGG-16. 
            
            input: feature map of dimension 16x16x128
            output: feature map of dimension 16x16x256
        """
        
        with tf.variable_scope("block_3",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            ## first convolution:
            conv_1 = self.conv2d(self.block_2,self.VGG16_weights[8],
                            self.VGG16_weights[9],"conv_1")
            
            relu_1 = tf.nn.relu(conv_1)
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_1)
            
            drop_1 = tf.nn.dropout(gnorm_1, self.prob)
            
            ## second convolution:
            conv_2 = self.conv2d(drop_1,self.VGG16_weights[10],
                            self.VGG16_weights[11],"conv_2")
            
            relu_2 = tf.nn.relu(conv_2)
            
            gnorm_2 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_2)
            
            drop_2 = tf.nn.dropout(gnorm_2, self.prob)
            
            ## third convolution:
            conv_3 = self.conv2d(drop_2,self.VGG16_weights[12],
                            self.VGG16_weights[13],"conv_3")
            
            relu_3 = tf.nn.relu(conv_3)
            
            gnorm_3 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_3)
            
            drop_3 = tf.nn.dropout(gnorm_3, self.prob)
            
            # max pooling:            
            pool_1 = tf.nn.max_pool(drop_3,ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],padding='SAME')
        
        return pool_1
    
    def block_4(self):
        """
            Fourth block of VGG-16 with max-pooling layer removed. 
            
            input: feature map of dimension 28x28x256
            output: feature map of dimension 28x28x512
        """
        
        with tf.variable_scope("block_4",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            ## first convolution:
            conv_1 = self.conv2d(self.block_3,self.VGG16_weights[14],
                            self.VGG16_weights[15],"conv_1")
            
            relu_1 = tf.nn.relu(conv_1)
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_1)
            
            drop_1 = tf.nn.dropout(gnorm_1, self.prob)
            
            ## second convolution:
            conv_2 = self.conv2d(drop_1,self.VGG16_weights[16],
                            self.VGG16_weights[17],"conv_2")
            
            relu_2 = tf.nn.relu(conv_2)
            
            gnorm_2 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_2)
            
            drop_2 = tf.nn.dropout(gnorm_2, self.prob)
            
            ## third convolution:
            conv_3 = self.conv2d(drop_2,self.VGG16_weights[18],
                            self.VGG16_weights[19],"conv_3")
            
            gnorm_3 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(conv_3)
            
            relu_3 = tf.nn.relu(gnorm_3)            
            
        
        return relu_3
    
    def block_5(self):
        """
            Fifth block of VGG-16. 
            
            input: feature map of dimension 16x16x512
            output: feature map of dimension 8x8x512
        """
        
        with tf.variable_scope("block_5",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            ## first convolution:
            conv_1 = self.conv2d(self.block_4,self.VGG16_weights[20],
                            self.VGG16_weights[21],"conv_1")
            
            relu_1 = tf.nn.relu(conv_1)
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_1)
            
            drop_1 = tf.nn.dropout(gnorm_1, self.prob)
            
            ## second convolution:
            conv_2 = self.conv2d(drop_1,self.VGG16_weights[22],
                            self.VGG16_weights[23],"conv_2")
            
            relu_2 = tf.nn.relu(conv_2)
            
            gnorm_2 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_2)
            
            drop_2 = tf.nn.dropout(gnorm_2, self.prob)
            
            ## third convolution:
            conv_3 = self.conv2d(drop_2,self.VGG16_weights[24],
                            self.VGG16_weights[25],"conv_3")
            
            relu_3 = tf.nn.relu(conv_3)
            
            gnorm_3 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(relu_3)
            
            drop_3 = tf.nn.dropout(gnorm_3, self.prob)
            
            # max pooling:
            pool_1 = tf.nn.max_pool(drop_3,ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],padding='SAME')
        
        return pool_1
    
    def concatenate_blocks(self):
        """
            Transformed and concatenated feature maps of the second, third, fourth and fifth 
            VGG-166 blocks.
            
            input: feature maps of the second, third, fourth and fifth VGG-166 blocks
            output: concatenated feature map of dimension 28x28x512
        """
        
        with tf.variable_scope("concatenate",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            ## first convolution:
            pool_1 = tf.layers.max_pooling2d(inputs=self.block_2, pool_size=[2, 2], strides=2)
            conv_1 = tf.layers.conv2d(inputs = pool_1,filters=128,
                                    padding="same",
                                    kernel_size=[3, 3],
                                    activation=tf.nn.relu,name="conv_1")
            
            ## second convolution:
            conv_2 = tf.layers.conv2d(inputs = self.block_3,filters=128,
                                    padding="same",
                                    kernel_size=[3, 3],
                                    activation=tf.nn.relu,name="conv_2")
            
            ## third convolution:
            conv_3 = tf.layers.conv2d(inputs = self.block_4,filters=128,
                                    padding="same",
                                    kernel_size=[3, 3],
                                    activation=tf.nn.relu,name="conv_3")
            
            ## fourth convolution:
            resize = tf.image.resize_images(images=self.block_5,size=[28,28]) ## bilinear upscaling
            conv_4 = tf.layers.conv2d(inputs = resize,filters=128,
                                    padding="same",
                                    kernel_size=[3, 3],
                                    activation=tf.nn.relu,name="conv_4")
            
            ## concatenate blocks:
            fused_blocks = tf.concat(values=[conv_1,conv_2,conv_3,conv_4],axis=3)
        
        
        return fused_blocks
    
    def image_representation(self):
        """
            An image representation corresponding to an input image.
            
            input:  concatenated feature map of dimension 28x28x512
            output: image representation of dimension 28x28x128
        """
        
        with tf.variable_scope("representation",reuse=tf.AUTO_REUSE):
            
            tf.set_random_seed(self.seed)
        
            conv = tf.layers.conv2d(inputs = self.fused_block,filters=128,
                                    padding="same",
                                    kernel_size=[3, 3],
                                    activation=tf.nn.relu)
        
        return conv
    
    def edges_and_vertices(self):
        """
            A model that predicts all vertices of an object within an image using 
            a similar architecture to the image encoder model. 
            
            input: RGB image of dimension 224x224x3
            output: downsampled vertex predictions of dimension 28x28x1
        """
        
        with tf.variable_scope("all_vertices",reuse=tf.AUTO_REUSE):

            ## boundary prediction:
            conv_1 = tf.layers.conv2d(inputs=self.representation,filters=32,
                                                padding="same",
                                                kernel_size=[3, 3],
                                                name="conv_1")
            
            gnorm_1 = GroupNormalization(groups=32, axis=-1, epsilon=0.1)(conv_1)
            
            relu_1 = tf.nn.relu(gnorm_1)
            
            flat = tf.layers.Flatten()(relu_1)
            
            E = tf.layers.dense(inputs=flat,units=784,activation=tf.nn.sigmoid)
                        
            E_plus_flat = tf.concat([flat,E],axis=1) ## combine image representation and boundary output
            
            V = tf.layers.dense(inputs=E_plus_flat,units=784,activation=tf.nn.sigmoid)  
            
            ## get the index associated with the first vertex:
            vertex_index = tf.argmax(V,axis=1)
            
            ## one-hot encode the first vertex:
            vertex_encoding = tf.one_hot(indices=vertex_index,depth=784)
                
        return tf.reshape(E,[-1,28,28,1]), tf.reshape(V,[-1,28,28,1]),vertex_encoding
