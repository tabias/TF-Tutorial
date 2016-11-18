import tensorflow as tf
import input_data

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[784, 500], initializer=xavier_init(784,500))
W2 = tf.get_variable("W2", shape=[500, 256], initializer=xavier_init(500,256))
W3 = tf.get_variable("W3", shape=[256, 128], initializer=xavier_init(256,128))
W4 = tf.get_variable("W4", shape=[128, 10], initializer=xavier_init(128,10))

b1 = tf.Variable(tf.zeros([500]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([128]))
b4 = tf.Variable(tf.zeros([10]))

L2 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W2),b2))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W3),b3))
hypothesis = tf.add(tf.matmul(L4, W4), b4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

training_epoch = 15
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            session.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # Compute average loss

        # show logs per epoch step
        avg_cost += session.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
        if epoch % display_step == 0:  # Softmax
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (session.run(b4))


    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

