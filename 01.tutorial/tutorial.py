import tensorflow as tf

ph1 = tf.placeholder(tf.int16)
ph2 = tf.placeholder(tf.int16)

session = tf.Session()

add = tf.add(ph1, ph2)
mul = tf.mul(ph1, ph2)

print(session.run(add, feed_dict={ph1:3, ph2:5}))
print(session.run(mul, feed_dict={ph1:2, ph2:6}))



