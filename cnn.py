__author__ = 'saideeptalari'
import tensorflow as tf
from utils.dataset import load_dataset,encode
from sklearn.model_selection import train_test_split
import numpy as np

batch_size = 128
"""
trainData,trainLabels = load_dataset("dataset/mnist_train.csv")
testData,testLabels = load_dataset("dataset/mnist_test.csv")
"""
Data,Labels = load_dataset("dataset/mnist_test.csv")
trainData,testData,trainLabels,testLabels = train_test_split(Data,Labels)

trainData = trainData.astype("float32")
testData = testData.astype("float32")

trainLabels = encode(trainLabels)
testLabels = encode(testLabels)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)
keep_prob1 = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3,3,32,32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.nn.dropout(h_pool2,keep_prob1)

W_fc1 = weight_variable([7*7*32,128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([128,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
train_step = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(12):
        costs = []
        #MiniBatch descent
        for start, end in zip(range(0, len(trainData), batch_size), range(batch_size, len(trainData)+1, batch_size)):
            _,c = sess.run([train_step,cross_entropy],feed_dict={x: trainData[start:end],y_: trainLabels[start:end],keep_prob: 0.5,keep_prob1:0.75})

            costs.append(c)
        c = np.mean(np.array(costs))
        acc = sess.run(accuracy,feed_dict={x: testData,y_: testLabels,keep_prob: 1.0,keep_prob1: 1.0})
        print "Epoch: {}, Cost: {}, Accuracy: {}".format(i+1,c,acc)

print "[INFO] optimization finished"















