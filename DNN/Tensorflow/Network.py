# coding=utf-8
import  numpy as np
import random
class Network(object):
    def __init__(self,size):
        self.num_layers = len(size)
        self.sizes = size
        self.biases = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
size = [2,3,1]
print(size[1:])
net = Network([2,3,1])
print(net.num_layers)
def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
    if test_data:n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)#随机打乱
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]
        for mini_batche in mini_batches:
            self.update_mini_batch(mini_batche,eta)
        if test_data:
            print("Epoch (0) : (1) / (2)".format(j,self.evaluate(test_data),n_test))
        else:
            print("Epoch {0} complete".format(j))