# coding=utf-8
import tensorflow as tf
import numpy as np
def layer(inputs,input_size,output_size,activation_function=None):
    W = tf.Variable(tf.random_normal([input_size,output_size]))
    bias = tf.Variable(tf.zeros([1,output_size])+0.1)
    W_bias_mul = tf.matmul(inputs,W) + bias
    if activation_function == None:
        outputs=W_bias_mul
    else:
        outputs=activation_function(W_bias_mul)
    return outputs
#创建数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
#输入数据
x_input = tf.placeholder(tf.float32,[None,1])
y_input = tf.placeholder(tf.float32,[None,1])
#添加网络层
#lay_1
l1=layer(x_input,1,10,activation_function=tf.nn.relu)
#lay_2
l2 = layer(l1,10,10,activation_function=tf.nn.relu)
#lay_2
l3 = layer(l2,10,5,activation_function=tf.nn.relu)
#out_layer
out_layer = layer(l3,5,1,activation_function=None)
#loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_input - out_layer),
                     reduction_indices=[1]))
#train
train_step=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess=tf.Session()
init = tf.global_variables_initializer()  # 替换成这样就好
sess.run(init)
#xunlian
for i in range(1000):
    sess.run(train_step,feed_dict={x_input:x_data,y_input:y_data})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={x_input:x_data,y_input:y_data}))
