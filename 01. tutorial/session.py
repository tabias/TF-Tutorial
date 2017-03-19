# import tensorflow library
import tensorflow as tf

# Constant string 'hello world'
# When you declare the variables, you can tf.Variable()
hello = tf.constant('hello world')

# This is very important concept
# TensorFlow has a data flow graph. so, nodes are connected each other
# In order to call the variable, constants in TF, we should declare the tf.Session()
# Within the session, evaluate the graph to fetch the value of constant below !
session = tf.Session()
print session.run(hello)