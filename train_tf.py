__author__ = 'saideeptalari'
import tensorflow as tf
from utils.dataset import load_dataset,plot_dataset,encode
import numpy as np

trainPath = "dataset/mnist_train.csv"
testPath = "dataset/mnist_test.csv"

trainData,trainLabels = load_dataset(trainPath)
testData,testLabels = load_dataset(testPath)

trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainLabels = encode(trainLabels)
testLabels = encode(testLabels)

trainData /= 255
testData /= 255

num_input = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_output = 10

num_epochs = 20

X = tf.placeholder("float",(None,num_input),name="X")
y = tf.placeholder("float",(None,num_output),name="y")

weights = {
    "W1": tf.Variable(tf.random_normal((num_input,num_hidden_1)),name="W1"),
    "W2": tf.Variable(tf.random_normal((num_hidden_1,num_hidden_2)),name="W2"),
    "W3": tf.Variable(tf.random_normal((num_hidden_2,num_output)),name="W3")
}

biases = {
    "b1": tf.Variable(tf.zeros((num_hidden_1)),name="b1"),
    "b2": tf.Variable(tf.zeros((num_hidden_2)),name="b2"),
    "b3": tf.Variable(tf.zeros((num_output)),name="b3"),
}


def feedforward(X,weights,biases):
    #layer 1
    Z1 = tf.matmul(X,weights["W1"]) + biases["b1"]
    a1 = tf.nn.relu(Z1)

    #layer 2
    Z2 = tf.matmul(a1,weights["W2"]) + biases["b2"]
    a2 = tf.nn.relu(Z2)

    #layer 3
    Z3 = tf.matmul(a2,weights["W3"] + biases["b3"])

    return Z3

pred = feedforward(X,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

for i in xrange(num_epochs):
    costs = []
    for start, end in zip(range(0, len(trainData), 100), range(100, len(trainData)+1, 100)):
        _,c = sess.run([optimizer,cost],feed_dict={X: trainData[start:end],y: trainLabels[start:end]})
        costs.append(c)
    c = np.mean(np.array(costs))
    p = sess.run(pred,feed_dict={X: testData,})
    acc = np.mean(np.equal(np.argmax(p,1),np.argmax(testLabels,1)))
    print "Epoch: {}, Cost: {}, Accuracy: {}".format(i+1,c,acc)

print "[INFO] optimization finished"

saver.save(sess,"output/tf_model.sess")
sess.close()