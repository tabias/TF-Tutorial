#LOGISTIC REGRESSION

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x = xy[0:-1]
y = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x)], -1.0, 1.0))
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# y = 0, h(x) =0 --> -log(1-h(x)) && y =1, h(x)=1 --> -log(h(x))
cost = tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))

LR = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(LR)
trained_data = optimizer.minimize(cost)

session = tf.Session()
init = tf.initialize_all_variables()
session.run(init)


for step in xrange(2001):
    session.run(trained_data, feed_dict={X:x, Y:y})


print session.run(hypothesis, feed_dict={X:[[1], [3], [4]]}) > 0.5





