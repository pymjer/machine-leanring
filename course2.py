import tensorflow as tf
import numpy as np


m1 = tf.constant([[3.,3.]]) 
m2 = tf.constant([[1.],[1.]])
product = tf.matmul(m1,m2)

session = tf.Session()

result = session.run(product)

print(result)