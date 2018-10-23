# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:29:38 2018

@author: Administrator
"""

import tensorflow as tf
"引入input_data.py,注：Python文件必须与input_data.py在同一文件夹下"
from tensorflow.examples.tutorials.mnist import input_data
def myprint(v):
    print(v)
    print(type(v))
    try:
        print(v.shape)
    except:
        try:
            print(len(v))
        except:
            pass


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./input_data', one_hot=True, validation_size=100)
    myprint(mnist.train.labels)
    myprint(mnist.validation.labels)
    myprint(mnist.test.labels)
    myprint(mnist.train.images)
    myprint(mnist.validation.images)
    myprint(mnist.test.images)
print("Training data size:", mnist.train.num_examples)
"x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。"
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
"建立模型"
y = tf.nn.softmax(tf.matmul(x,W) + b)
"输入正确值"
y_ = tf.placeholder("float", [None,10])
"计算交叉熵"
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
"用梯度下降算法训练模型"
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"评估模型"
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))