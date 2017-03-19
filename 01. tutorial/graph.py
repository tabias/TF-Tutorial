import tensorflow as tf

# This tf.Graph() concept. By using tf.Graph(), we can save computation in distributed system.
# More detailed explanation will be later
g = tf.Graph()
with g.as_default():
    x = tf.add(3,5)

sess = tf.Session(graph=g)
sess.run(x)