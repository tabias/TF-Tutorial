import tensorflow as tf

# tf.placeholder is also very useful concept in TF
# In placeholder we don't have to give a value to it directly,
# we can change the value by using feed_dict={a:value1, b:value2}
ph1 = tf.placeholder(tf.int16)
ph2 = tf.placeholder(tf.int16)

# session
session = tf.Session()

# tf.add(a,b) --> return a+b, tf.mul(a,b) --> return a*b
add = tf.add(ph1, ph2)
mul = tf.multiply(ph1, ph2)

# run the session with different values(by using tf.placeholder)
print(session.run(add, feed_dict={ph1:3, ph2:5}))
print(session.run(mul, feed_dict={ph1:2, ph2:6}))



