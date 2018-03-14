# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#hyperparameters
lr = 0.001#learning rate
training_iters = 100000# train step 上限
batch_size = 128
n_inputs =28
n_steps = 28
n_hidden_units = 128 #neurons in hidden layer
n_classes = 10 #MNIST classes
#x,y placeholder
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])
#对weights biases 初始值的定义
weights={
    #shape(28,128)
    "in":tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    "out":tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
def RNN(X,weights,biases):
    #3维变2维
    X=tf.reshape(X,[-1,n_inputs])
    #X_in = W*X +b
    X_in = tf.matmul(X,weights["in"]) +biases["in"]
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=0.1,state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1],weights["out"]) + biases["out"]
    return results
pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1