import tensorflow as tf

#sudo apt-get install python-matplotlib
import matplotlib.pyplot as plt

# dataset
X = [1., 2., 3.]
Y = [1., 2., 3.]

# define hypothesis
m = n_samples = len(X)
W = tf.placeholder(tf.float32)
hypothesis = tf.mul(X, W)

# define cost function, initiate session
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / (m)

# TF-VERSION 1.0 CHANGED METHOD
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# for graph
W_val = []
cost_val = []

# draw graph(-3.0 to 5.0)
for i in range(-30, 50):
    print i*0.1, session.run(cost, feed_dict={W: i*0.1})
    W_val.append(i*0.1)
    cost_val.append(session.run(cost, feed_dict={W: i*0.1}))

# show Cost value(global minimum)
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()

