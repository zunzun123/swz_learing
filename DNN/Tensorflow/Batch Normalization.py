# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#parameters
ACTIVATION = tf.nn.relu # 每一层都使用 relu
N_LAYERS = 7            # 一共7层隐藏层
N_HIDDEN_UNITS = 30     # 每个层隐藏层有 30 个神经元
#bulit net
def built_net(xs,ys,norm):
    def add_layer(inputs,in_size,out_size,activation_function=None):
        #添加层功能
        Weights = tf.Variable(tf.random_normal([in_size,out_size],mean=0,stddev=0.1))
        biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    fix_seed(1)

    layers_inputs = [xs]#记录每层的input
    #loop建立所有层
    for i in range(N_LAYERS):
        layers_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value
