__author__ = 'saideeptalari'

import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path,delimiter=","):
    arr = np.loadtxt(path,delimiter=delimiter,dtype="uint8")
    labels = arr[:,0]
    data = arr[:,1:]
    #data = data.reshape(-1,28,28)
    return data,labels

def plot_dataset(data,labels,predictions=None,annotate=True):
    assert len(data)==len(labels) == 16

    for i in xrange(len(labels)):
        plt.subplot(4,4,i+1)
        plt.axis("off")
        plt.imshow(data[i],cmap="gray")
        title = "True: {}".format(labels[i])
        if predictions is not None:
            title += " Predicted: {}".format(predictions[i])
        if annotate:
            plt.title(title)
    plt.show()

def encode(y):
    Y = np.zeros((y.shape[0],len(np.unique(y))))
    for i in xrange(y.shape[0]):
        Y[i,y[i]] = 1
    return Y