# Neural network

import tensorflow as tf
import numpy as np

# load x1, x2 data to x_data, y data to y_data
# it should be compatible to compute
xy = np.loadtxt('train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# define w1, w2 cause process layer2 NN
W1 = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))

# define bias
b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1)+b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2)+b2)

cost = tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.initialize_all_variables()

# training data, get accuracy
with tf.Session() as session:
    session.run(init)

    for step in xrange(2001):
        session.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 ==0:
            print session.run(cost, feed_dict={X:x_data, Y:y_data})


    # tf,floor(hypothesis+0.5) value should be 0 or 1
    # accuracy can get from answer
    answer = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))
    print session.run([hypothesis], feed_dict={X:x_data, Y:y_data})
    print "Accuracy : ", accuracy.eval({X:x_data, Y:y_data})*100, "%"


