__author__ = 'saideeptalari'
import tensorflow as tf
import numpy as np
from utils.dataset import load_dataset,encode

#define hyperparameters
batch_size = 100
nb_epochs = 12
learning_rate = 0.001
strides = 1
ksize = 2


#load training and testing sets
trainData,trainLabels = load_dataset("dataset/mnist_train.csv")
testData,testLabels = load_dataset("dataset/mnist_test.csv")

#convert to float
trainData = trainData.astype("float32")
testData = testData.astype("float32")

#one-hot encode labels
trainLabels = encode(trainLabels)
testLabels = encode(testLabels)

#normalize data
trainData = trainData/255
testData = testData/255


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
pool2 = tf.nn.dropout(pool2,keep_prob=0.25)

#Fully-Connected Layer-1
fc1 = tf.reshape(pool2,shape=[-1,weights["W3"].get_shape().as_list()[0]])
fc1 = tf.nn.bias_add(tf.matmul(fc1,weights["W3"]),biases["b3"])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1,keep_prob=0.5)

#Fully-Connected Layer-2
fc2 = tf.nn.bias_add(tf.matmul(fc1,weights["W4"]),biases["b4"])

#pred = tf.nn.softmax(fc2)

#define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2,y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)


for i in xrange(nb_epochs):
    costs = []
    #MiniBatch descent
    for start, end in zip(range(0, len(trainData), batch_size), range(batch_size, len(trainData)+1, batch_size)):
        _,c = sess.run([optimizer,cost],feed_dict={X: trainData[start:end],y: trainLabels[start:end]})
        costs.append(c)
    c = np.mean(np.array(costs))
    p = sess.run(fc2,feed_dict={X: testData,y: testLabels})
    acc = np.equal(np.argmax(p,axis=1),np.argmax(testLabels,axis=1))
    acc = np.mean(acc)
    print "Epoch: {}, Cost: {}, Accuracy: {}".format(i+1,c,acc)

print "[INFO] optimization finished"

sess.close()