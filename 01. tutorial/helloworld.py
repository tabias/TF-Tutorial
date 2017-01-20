import tensorflow as tf
hello = tf.constant('hello world')
session = tf.Session()
print session.run(hello)