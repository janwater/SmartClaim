

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:11:02 2017

@author: E154709
"""
import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# functions
def _create_filename(path, fname, affix = '.csv'):
    # create new file name
    output_name = fname + affix        
    foutput_name = os.path.join(path, '', output_name)
             
    return foutput_name

def _read_inputTags():
    # read input tags
    inputTags = ['numClaims',
               'numSales',
               'dollarPerClaim',
               'dollarPerSale',
               'claimRatio',
               'INVENTORY_STYLE_CD',
               'INVENTORY_SIZE_CD',
               'INVENTORY_BACKING_CD',
               'INVENTORY_COLOR_CD'
              ]
    return inputTags

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data = np.asarray(data)
    labels = np.asarray(labels)
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_mlp(n_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(_):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    # set working directory
    parentDir = 'C:/Users/E154709/Desktop/Mohawk/Claims'
    os.chdir(parentDir)
    #modeling data
    dataPath = parentDir + '/Data/'
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    
    #create data file
    fname = 'modelData2'
    fmodelData = _create_filename(dataPath,fname)
    modelData = pd.read_csv(fmodelData, encoding='latin1') # Read the data
    
    # import input tags
    inputTags = _read_inputTags()
    yTags = 'Claims'
    
    # read model data
    X = modelData[inputTags]
    y = modelData[yTags]
    
    # split data
    ts = 0.4
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, stratify=y, test_size=ts, random_state=531) 
    yTrn_binary = to_categorical(yTrain)
    yTst_binary = to_categorical(yTest)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 9])
    W = tf.Variable(tf.zeros([9, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        #batch_xs, batch_ys = xTrain.next_batch(100)
        batch_xs, batch_ys = next_batch(50, xTrain, yTrn_binary)
        #print(batch_xs.shape, batch_ys.shape)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: np.asarray(xTest),
                                        y_: np.asarray(yTst_binary)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='C:/Users/E154709/Desktop/Mohawk/Claims',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)