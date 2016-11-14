# softmax classification

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x = np.transpose(xy[0:3])
y = np.transpose(xy[3:])

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    for step in xrange(2001):
        session.run(optimizer, feed_dict={X:x, Y:y})
        if step % 200 ==0:
            print session.run(W)