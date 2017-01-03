'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
strides = 1
ksize = 2

#Initialize placeholders
X = tf.placeholder(dtype="float",shape=(None,784))
y = tf.placeholder(dtype="float",shape=(None,10))

#define weights and biases
weights = {'W1': tf.Variable(tf.random_normal([3,3,1,32])),
           'W2': tf.Variable(tf.random_normal([3,3,32,32])),
           'W3': tf.Variable(tf.random_normal([5*5*32,128])),
           'W4': tf.Variable(tf.random_normal([128,10]))
           }

biases = {"b1": tf.Variable(tf.random_normal([32])),
          "b2": tf.Variable(tf.random_normal([32])),
          "b3": tf.Variable(tf.random_normal([128])),
          "b4": tf.Variable(tf.random_normal([10]))
          }

x = tf.reshape(X,shape=[-1,28,28,1])

#First convolution Layer
conv1 = tf.nn.conv2d(x,weights["W1"],strides=[1,strides,strides,1],padding="VALID")
conv1 = tf.nn.bias_add(conv1,biases["b1"])
conv1 = tf.nn.relu(conv1)
#maxpooling
pool1 = tf.nn.max_pool(conv1,ksize=[1,ksize,ksize,1],strides=[1,2,2,1],padding="VALID")

#Second convolution Layer
conv2 = tf.nn.conv2d(pool1,weights["W2"],strides=[1,strides,strides,1],padding="VALID")
conv2 = tf.nn.bias_add(conv2,biases["b2"])
conv2 = tf.nn.relu(conv2)
#maxpooling
pool2 = tf.nn.max_pool(conv2,ksize=[1,ksize,ksize,1],strides=[1,2,2,1],padding="VALID")
pool2 = tf.nn.dropout(pool2,keep_prob=0.75)

#Fully-Connected Layer-1
fc1 = tf.reshape(pool2,shape=[-1,weights["W3"].get_shape().as_list()[0]])
fc1 = tf.nn.bias_add(tf.matmul(fc1,weights["W3"]),biases["b3"])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1,keep_prob=0.5)

#Fully-Connected Layer-2
fc2 = tf.nn.bias_add(tf.matmul(fc1,weights["W4"]),biases["b4"])

#pred = tf.nn.softmax(fc2)
pred = fc2
#define cost


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))