__author__ = 'saideeptalari'

import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path,delimiter=","):
    """
    Loads dataset from a given path and delimiter

    :param path: Path to dataset that needed to be loaded
    :param delimiter: delimiter (default=",")
    :return: dataset,labels
    """
    arr = np.loadtxt(path,delimiter=delimiter,dtype="uint8")
    labels = arr[:,0]
    data = arr[:,1:]
    #data = data.reshape(-1,28,28)
    return data,labels

def plot_dataset(data,labels,predictions=None,annotate=True):
    """
    Plots sample images given of shape (16,h,w)

    :param data: array of images of shape (16,h,w) where h:height,w:width of the image
    :param labels: true labels associated with the images
    :param predictions: (optional) if given it prints the predicted labels on the images
    :param annotate: (default=True) whether to annotate labels or not
    :return:
    """
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
    """
    One-hot encodes the labels
    :param y: labels to be encoded
    :return: one-hot encoded labels
    """
    Y = np.zeros((y.shape[0],len(np.unique(y))))
    for i in xrange(y.shape[0]):
        Y[i,y[i]] = 1
    return Y