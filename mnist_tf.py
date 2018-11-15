from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Tensorflow has the mnist data builtin
data_dir = '/tmp/tensorflow/mnist/input_data'

# Import data
mnist = input_data.read_data_sets(data_dir,one_hot=True)

n = 784  # Number of input features
N = 10   # Number of classes

# X is the vector of inputs (though it's just a placeholder until a tensorflow session is started)
X = tf.placeholder("float", [None, n])

# y is the vector of targets
y = tf.placeholder("float", [None, N])

# Create the model
# layer 1 weights and biases
W_0 = tf.Variable(tf.random_normal([n,N],stddev=0.0))
b_0 = tf.Variable(tf.random_normal([N],stddev=0.0))

# Create neural network
def perceptron(x):
    out_layer = tf.add(tf.matmul(x,W_0),b_0)
    return out_layer

# define prediction object
y_pred = perceptron(X)

# Define loss function (combined softmax and cross-entropy output) 
loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

gamma = 1.0
regularization_op = gamma*tf.reduce_sum(abs(W_0)**2)

# Specify learning rate
learning_rate = 0.001

# Define optimization step
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# The optimization procedure (minimizing the softmax cross_entropy)
train_op = optimizer.minimize(loss_op + regularization_op)

# Initialize all the variables and the session 
# (tensorflow doesn't compute variable values unless run by as session)
sess = tf.InteractiveSession()
init  = tf.global_variables_initializer().run()

# Train
N_iterations = 2500
sample_size = 256

for i in range(N_iterations):

    # Pull a sample from the training set
    batch_xs, batch_ys = mnist.train.next_batch(sample_size)

    # Run tensor flow objects: train_op updates the weights, loss_op compute the cost function
    _,c = sess.run([train_op,loss_op], feed_dict={X: batch_xs, y: batch_ys})

    # Print statistics every 1000 steps
    if i%1000==0:
        # Test trained model
        pred = tf.nn.softmax(y_pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, y: mnist.test.labels}),c)

# You can acquire the values of your layer weights with
w = sess.run(W_0)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = [12,12]
fig,axs = plt.subplots(nrows=2,ncols=5)
fig.subplots_adjust(wspace=0.02,hspace=0.02)
for i in range(2):
    for j in range(5):
        axs[i,j].imshow(w[:,5*i+j].reshape((28,28)),interpolation='bilinear')
        axs[i,j].set_title(str(5*i+j))



