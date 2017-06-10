import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder(tf.float32, [None,10])

## 计算交叉熵
## 问题，y_ 和 tf.log(y) 两个都不是基本类型，为何可以用 * 来运算？不是应该用 tf.matmul来计算吗？
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

## 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  if i % 100 == 0:
  	print("进度：" , (i / 1000))

  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

## 正确值,正确为1，错误为0 [1,0,0,1,1]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

## 取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("正确率：" ,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))