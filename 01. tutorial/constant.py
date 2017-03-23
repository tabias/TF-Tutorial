import tensorflow as tf


# constant whose name is vector
a = tf.constant([2, 2], name="vector")

# fill with zero
b = tf.zeros([2, 3], tf.int32)

# make all zero
c = tf.constant([[1,2], [3,4], [5,6]], name="zzz")
d = tf.zeros_like(c)

# fill ten with [4,5] vector
e = tf.fill([4, 5], 10)

# number sequence
f = tf.linspace(5.0, 10.0, 5, name="linspace")

# range settings
g = tf.range(5, 30, 5)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(d)
    print sess.run(e)
    print sess.run(f)
    print sess.run(g)