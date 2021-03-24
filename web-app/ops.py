import numpy as np 
import tensorflow as tf

def lrelu(x, leak=0.02):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope):
    shape = input_.get_shape()
    with tf.variable_scope(scope):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [output_size], initializer=tf.zeros_initializer())
        print('linear', 'in', shape, 'out', (shape.as_list()[0], output_size))
        return tf.matmul(input_, matrix) + bias
