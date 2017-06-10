import tensorflow as tf
import numpy as np

sess = tf.Session()
m = sess.run(tf.truncated_normal((5,10), stddev = 0.1) )
print(type(m))
print(m)

#使用tensorflow中的tf.argmax()
col_max = sess.run(tf.argmax(m, 0) )  #当axis=0时返回每一列的最大值的位置索引
print("axis=0,返回每一列的最大值索引：",col_max)

row_max = sess.run(tf.argmax(m, 1) )  #当axis=1时返回每一行中的最大值的位置索引
print("axis=1,返回每一行的最大值索引：",row_max)

# 打印输出
# array([2, 3, 0, 3, 0, 0, 0, 0, 3, 4])
# array([5, 0, 0, 8, 9])

#使用numpy中的numpy.argmax
row_max = m.argmax(0)
print(row_max)

col_max = m.argmax(1)
print(col_max)

# 打印输出
# array([2, 3, 0, 3, 0, 0, 0, 0, 3, 4])
# array([5, 0, 0, 8, 9])