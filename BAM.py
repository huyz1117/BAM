# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:25:53 2019

@author: huyz

Reference: [BMVC2018] BAM: Bottleneck Attention Module
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

batch_norm_params = {
                    # Decay for moving averages
                    'decay': 0.995,
                    # epsilon to prevent 0 in variance
                    'epsilon': 0.001,
                    # force in-place updates of mean and variances estimates
                    'updates_collection': None,
                    # moving averages ends up in the trainable variables collection
                    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]}
                    
def BAM(inputs, batch_norm_params, reduction_ratio=16, dilation_value=4, reuse=None, scope'BAM'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d], activation=None):
                
                input_channel = inputs.get_shape().as_list()[-1]
                num_squeeze = input_channel // reduction_ratio
                
                # Channel attention
                gap = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
                channel = slim.fully_connected(gap, num_squeeze, activation_fn=None, scope='fc1')
                channel = slim.fully_connected(fc1, input_channel, activation_fn=None,
                normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='fc2')
                
                # Spatial attention
                spatial = slim.conv2d(inputs, num_squeeze, 1, padding='SAME', scope='conv1')
                spatial = slim.repeat(spatial, 2, slim.conv2d, num_squeeze, 3, padding='SAME', rate=dilation_value, scope='conv2')
                spatial = slim.conv2d(spatial, 1, 1, padding='SAME', scope='conv3',
                                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
                                    
                # combined two attention branch
                combined = tf.nn.sigmoid(channel + spatial)
                
                output = inputs + inputs * combined
                
                return output
                
                