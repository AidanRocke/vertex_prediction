#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:14:15 2018

@author: aidanrocke
"""

import numpy as np
import unittest
import tensorflow as tf

## import additional modules:
from image_encoder import image_encoder

def encoder_inference():
    
    random_seed = 42
    
    tf.reset_default_graph()
    
    encoder = image_encoder(random_seed)
    
    input_ = np.zeros((1,224,224,3))
    
    with tf.Session() as sess:
        
        sess.run(encoder.init_g)
                    
        vertex = sess.run(encoder.vertex,feed_dict = {encoder.input_image:input_})
            
    return np.shape(vertex)
        

class MyTest(unittest.TestCase):
    def test_inference(self):
        self.assertEqual(encoder_inference(), (1,784))
        

if __name__ == '__main__':
    unittest.main()
        



