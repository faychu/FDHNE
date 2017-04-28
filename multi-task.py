__author__ = 'fay'
import tensorflow as tf
import numpy as np

# ======================
# Define the Graph
# ======================

# Define the placeholders
X = tf.placeholder('float', [10, 10], name='x')
Y1 = tf.placeholder('float', [10, 20], name='Y1')
Y2 = tf.placeholder('float', [10, 20], name='Y2')

# Define the weights for the layers
initial_shared_layer_weights = np.random.rand(10, 20)
initial_Y1_layer_weights = np.random.rand(20, 20)
initial_Y2_layer_weights = np.random.rand(20, 20)

shared_layer_weights = tf.Variable(initial_shared_layer_weights, name='share_w',dtype='float32')
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name='share_Y1', dtype='float32')
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name='share_Y2', dtype='float32')

# Construct the layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

# loss function
Y1_loss = tf.nn.l2_loss(Y1-Y1_layer)
Y2_loss = tf.nn.l2_loss(Y2-Y2_layer)
joint_loss = Y1_loss+Y2_loss

# optimizers
Optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(joint_loss)
Y1_op = tf.train.GradientDescentOptimizer(0.01).minimize(Y1_loss)
Y2_op = tf.train.GradientDescentOptimizer(0.01).minimize(Y2_loss)

# Joint Training
with tf.Session() as session:
    session.run(tf.global_variables_initializer())  # initialize
    _, joint_loss = session.run([Optimizer, joint_loss],
                                {
                                    X:  np.random.rand(10,10)*10,
                                    Y1: np.random.rand(10,20)*10,
                                    Y2: np.random.rand(10,20)*10
                                })
    print(joint_loss)