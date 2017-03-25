import tensorflow as tf

# session define
session = tf.Session()

# x, y data training set
x_data = [1,2,3]
y_data = [3,5,7]

# define w,b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# process placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# define hyopthesis
hypothesis = w*X+b

# linear regression cost function & process gradientDescent algorithm
cost = tf.reduce_mean(tf.square(hypothesis-y_data))
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# TF-VERSION 1.0 CHANGED METHOD
init = tf.global_variables_initializer()
session.run(init)


# find a value
for step in xrange(2001):
    session.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 ==0:
        print step, session.run(w), session.run(b)


# examine a value
print session.run(hypothesis, feed_dict={X:5})