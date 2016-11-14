import tensorflow as tf
import numpy as np

#load training data to array
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

# x is training data & y is result(binary classification)
x = xy[0:-1]
y = xy[-1]

# process placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# define W, h and hypothesis(Logistic Classification)
# sigmoid hypothesis = 1/1+e^h
W = tf.Variable(tf.random_uniform([1, len(x)], -1.0, 1.0))
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# cost function(logistic, can get global minimum using GradientDescent algorithm)
cost = -tf.reduce_mean(Y*tf.log(1+hypothesis)+(1-Y)*tf.log(1-hypothesis))

# minimizing cost function(cause' convex function)
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
trained_data = optimizer.minimize(cost)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# training matrix w
for step in xrange(2001):
    session.run(trained_data, feed_dict={X:x, Y:y})
    if step%100 == 0:
        print step, session.run(cost, feed_dict={X:x, Y:y}), session.run(W)


# examine classification
print '-----------------------------------------'
print session.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [3], [4]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [4], [4]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [4], [5]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [5], [4]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5
print session.run(hypothesis, feed_dict={X:[[1], [7], [7]]}) > 0.5


