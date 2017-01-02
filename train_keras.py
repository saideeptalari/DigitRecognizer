from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from utils.dataset import load_dataset,encode

#load training set and encode
trainData,trainLabels = load_dataset("dataset/mnist_train.csv")
trainLabels = encode(trainLabels)
#load testing set and encode
testData,testLabels = load_dataset("dataset/mnist_test.csv")
testLabels = encode(testLabels)
#convert to float
trainData = trainData.astype("float32")
testData = testData.astype("float32")
#normalize
trainData /= 255
testData /= 255

#create model
model = Sequential()
model.add(Dense(input_dim=784,output_dim=256,activation='relu',init="normal"))
model.add(Dense(output_dim=256,activation='relu',init="normal"))
model.add(Dense(output_dim=10,activation="softmax"))
#compile and fit
model.compile("adam","categorical_crossentropy",metrics=["accuracy"])
model.fit(trainData,trainLabels,batch_size=100,nb_epoch=25,verbose=2,validation_data=(testData,testLabels))

print(model.summary())
score = model.evaluate(testData,testLabels)
print('Test cost:', score[0])
print('Test accuracy:', score[1])

#save model to disk
model.save("output/mnist.model")
