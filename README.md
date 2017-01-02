# DigitRecognizer

A simple project that is created to support the blog post @ http://hackevolve.com/recognize-handwritten-digits-1 and http://hackevolve.com/recognize-handwritten-digits-2

It is simply an **Hand written** digit recognizer, It takes an image and recognizes the digits in them. This can be extended to recognizing characters as well.

![input/output](http://hackevolve.com/wp-content/uploads/2017/01/output.jpg)
## Files description

* `utils/dataset.py` - A simple utility that loads the dataset from disk, encodes, plots the dataset
* `recognize.py` - It is the driver script which takes an image and recognizes the digits in it. It takes two command line arguments namely --model and --image.
* `train_keras.py` - A simple script which trains a model on **MNIST** dataset and save to disk using keras.
* `train_tf.py` - It's just an tensorflow implementation of the above.
* `train_keras_cnn.py` - It trains the model on Convolution Neural Network and save the model to disk.
* `train_tf_cnn.py` - It's just an tensorflow implementation of the above.

### About the implemented CNN model

The below text represents the CNN model implemented in this project

            Input---> shape(-1,28,28,1)
            Convolution Layer 1 ---> filter_shape:(3,3), no.of.filters: 32, strides: 1, padding: valid
            Activation ---> Relu
            MaxPooling ---> pool_size: (2,2) i.e., halves the image, padding: valid
            
            Convolution Layer 2 ---> filter_shape:(3,3), no.of.filters: 32, strides: 1, padding: valid
            Activation ---> Relu
            MaxPooling ---> pool_size: (2,2) i.e., halves the image, padding: valid
            Dropout ---> 0.25 (keep probability)
            
            Fully Connected Layer 1 ---> num_units: 128,input_dim: 5*5*32
            Activation ---> Relu
            Dropout ---> 0.5 (keep probability)
            
            Fully Connected Layer 2 ---> num_units: 10, input_dim: 128
            Activation ---> Softmax
            
            Optimizer ---> Adadelta, learning_rate: 0.001
            fit ---> batch_size: 128, nb_epochs: 12


## Accuracy obtained
The accuracy obtained is more than **99%** using the Convolution Neural Network model 96% while using simple Multi Layer Perceptron.

![accuracy](http://hackevolve.com/wp-content/uploads/2017/01/Screenshot-from-2017-01-02-19-09-26.png)

