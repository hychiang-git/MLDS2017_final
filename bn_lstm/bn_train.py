# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from lstm import LSTMCell, BNLSTMCell, orthogonal_initializer

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('/tmp3/vicky/', one_hot=True)

# hyperparameters
lr = 0.001
batch_size = 128
training_iters = 1000*batch_size

#n_inputs = 28   # MNIST data input (img shape: 28*28)
#n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])
training = tf.placeholder(tf.bool)

# Define weights
weights = {
    # (28, 128)
    #'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    #'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    'out': tf.get_variable('w_out', [n_hidden_units, n_classes], initializer=orthogonal_initializer())
}
biases = {
    # (128, )
    #'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    #'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    'out': tf.get_variable('b_out', [n_classes, ])
    # (10, )
}


def RNN(X, weights, biases):
    X_in = tf.expand_dims(X, -1)
    cell = BNLSTMCell(n_hidden_units, training) #LSTMCell(hidden_size)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, state  = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    _, final_hidden = state
    results = tf.nn.softmax(tf.matmul(final_hidden, weights['out']) + biases['out'])    # shape = (128, 10)

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
            training: True
        })
        loss = sess.run(cost, feed_dict={x: batch_xs,y: batch_ys,training: False})
        if step % 20 == 0:
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            print 'epoch: {},loss: {},valid accuracy: {}'.format(step,loss,sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            training: False
            }))
        if step % 100 == 0:
            for v in tf.all_variables():
                print v.name
            print sess.run('rnn/BNLSTMCell/c_mean:0')
            print sess.run('rnn/BNLSTMCell/xh_mean:0')
        step += 1



