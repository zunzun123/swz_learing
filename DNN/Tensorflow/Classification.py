# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA",one_hot=True)
#网络层函数
def add_layer(inpus,in_size,out_size,activation_function = None,):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inpus,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs
#计算精确度
def computer_accurary(v_xs,v_ys):
    global prediction
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
#输入层
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
#输出层
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
# the error between prediction and real data
#交叉熵()
cross_entropy = 4(-tf.reduce_sum(ys * tf.log(prediction),
reduction_indices=[1]))
#训练
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
#初始化变量
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(computer_accurary(
mnist.test.images, mnist.test.labels))