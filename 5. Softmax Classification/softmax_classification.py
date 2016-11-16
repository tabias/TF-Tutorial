# SOFTMAX CLASSIFICATION(multinomial)

import tensorflow as tf
import numpy as np

# numpy !
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

# transpose matrix
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# placeholder
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

# initialize matrix
W = tf.Variable(tf.zeros([3, 3]))

# get hypothesis softmax
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# get cost function D(S,L) --> softmax algorithm(multinomial classification)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# gradientDescent algorithm
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
init = tf.initialize_all_variables()

# define session, train
with tf.Session() as session:
    session.run(init)

    for step in xrange(2001):
        session.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, session.run(cost, feed_dict={X:x_data, Y:y_data}), session.run(W)


    # it can be not accurate because training data has only 9 samples!
    a = session.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print a, session.run(tf.argmax(a, 1))

    b = session.run(hypothesis, feed_dict={X: [[1, 0, 1]]})
    print b, session.run(tf.argmax(b, 1))


