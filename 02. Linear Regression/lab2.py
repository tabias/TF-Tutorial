# tensor flow
import tensorflow as tf

# training data
x_data = [1,2,3]
y_data = [1,2,3]

# w means weight value, b menas bias value
# At first, these values are unknown. So, I made a random number
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# I made a hypothesis about the linear regression
# form is weight * x(input) + bias value
hypothesis = w*x_data + b

# get a cost by using cost function ( linear regression cost function )
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

# cost should be minimal if we get optimal result !
# So we use GradientDesent method which is basic
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
trainer = optimizer.minimize(cost)

# TF-VERSION 1.0 CHANGED METHOD
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# testing 2000 times, and computer knows weight value is 1, bias is also 1
# you can see the cost variations. It converges zero ( gradient descent algorithm )
for step in xrange(2001):
    sess.run(trainer)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w), sess.run(b)