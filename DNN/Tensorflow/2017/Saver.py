# coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#设置随机数种子
tf.set_random_seed(1)
np.random.seed(1)
#创建数据
x = np.linspace(-1,1,100)[:,np.newaxis]#shape(100,1)
noise = np.random.normal(0,0.1,size=x.shape)
#print(noise)
y = np.power(x,2) + noise#power 数组的元素分别求n次方,shape(100,1)
def save():
    print("This is save")
    #建立神经网络
    #s输入层
    tf_x = tf.placeholder(tf.float32,x.shape)
    tf_y = tf.placeholder(tf.float32,y.shape)
    #隐藏层
    hidden_layer = tf.layers.dense(tf_x,10,tf.nn.relu)
    #输出层
    out_layer = tf.layers.dense(hidden_layer,1)
    #计算cost
    loss = tf.losses.mean_squared_error(tf_y,out_layer)
    #训练
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
    sess = tf.Session()
    #初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)
    #define saver for saving and restore
    saver = tf.train.Saver()

    for step in range(100):
        sess.run(train_op,{tf_x:x,tf_y:y})
    saver.save(sess,"./saver_net/params",write_meta_graph=False)
    #plotting
    pred,l = sess.run([out_layer,loss],{tf_x:x,tf_y:y})
    plt.figure(1,figsize=(10,5))
    plt.subplot(121)#创建子图
    plt.scatter(x,y)
    plt.plot(x,pred,"r-",lw=5)
    plt.text(-1, 1.2, 'Save Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
def reload():
    print("This is reload")
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, y.shape)  # input y
    l_ = tf.layers.dense(tf_x, 10, tf.nn.relu)  # hidden layer
    o_ = tf.layers.dense(l_, 1)  # output layer
    loss_ = tf.losses.mean_squared_error(tf_y, o_)  # compute cost

    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, './saver_net/params')
    var_list=tf.trainable_variables()
    for idx ,v in enumerate(var_list):
        print("param {:3}: {:15} "
              "  {}".format(idx, str(v.get_shape()), v.name))
    # plotting
    pred, l = sess.run([o_, loss_], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()
save()
# destroy previous net
tf.reset_default_graph()

reload()

