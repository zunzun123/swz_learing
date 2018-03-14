# coding=utf-8
import tensorflow as tf
import numpy as np
from matplotlib import  pyplot as plt
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name="layer%s"%n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            #接下来,我们层中的Weights设置变化图, tensorflow中提供了tf.histogram_summary()方法,用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量
            tf.summary.histogram(layer_name+"/weights",Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            #tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs
#导入数据
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 +noise
#define placeholder for inputs to networl
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32,[None,1],name="x_input")
    ys = tf.placeholder(tf.float32,[None,1],name="y_input")
#搭建网络
#定义隐藏层
l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)
#输出层
prediction = add_layer(l1,10,1,2,activation_function=None)
#计算误差
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(ys-prediction),reduction_indices=[1]))
    #Loss 的变化图和之前设置的方法略有不同. loss是在tesnorBorad 的event下面的, 这是由于我们使用的是tf.scalar_summary() 方法.
    tf.summary.scalar("loss",loss)
#学习率
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#定义session
sess = tf.Session()
#写入文件
#接下来， 开始合并打包。 tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)#初始化所有变量
#画图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#训练
for i in range(1000):
    #training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(1)
        #以上这些仅仅可以记录很绘制出训练的图表，
        #  但是不会记录训练的数据。
        #  为了较为直观显示训练过程中每个参数的变化，我们每隔上50次就记录一次结果 , 同时我们也应注意, merged 也是需要run 才能发挥作用的,所以在for循环中写下：
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)