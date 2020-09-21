import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        pass   #TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        pass   #TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Model Function
def conv_net(data, weights, biases, training=False):
    # Convolution layers
    conv1 = conv2d(data, weights['c1'], biases['c1']) # [28,28,16]
    conv2 = conv2d(conv1, weights['c2'], biases['c2']) # [28,28,16]
    pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [14,14,16]

    conv3 = conv2d(pool1, weights['c3'], biases['c3']) # [14,14,32]
    conv4 = conv2d(conv3, weights['c4'], biases['c4']) # [14,14,32]
    pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [7,7,32]

    # Flatten
    flat = tf.reshape(pool2, [-1, weights['d1'].get_shape().as_list()[0]])
    # [7*7*32] = [1568]

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights['d1']), biases['d1']) # [128]
    fc1 = tf.nn.relu(fc1) # [128]

    # Dropout
    if training:
        fc1 = tf.nn.dropout(fc1, rate=0.2)

    # Output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) # [10]
    return out



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    xTrainP = xTrain.reshape((60000, 784))
    xTestP = xTest.reshape((10000, 784))
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain, = data
    xTest, yTest = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(str(xTrain.shape))
        print(str(yTrain.shape))


        model.fit(xTrain, yTrain, epochs=5)
        y_pred=model.predict_classes(xTest)

        yTestNew = np.argmax(yTest, axis=1)
        con_mat = tf.math.confusion_matrix(labels=yTestNew, predictions=y_pred).numpy()

        # Printing the result
        print('Confusion_matrix: ',con_mat)
        print("loss: %f\naccuracy: %f" % tuple(model.evaluate(xTest, yTest)))

        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    #xTest, yTest = data
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("loss: %f\naccuracy: %f" % tuple(model.evaluate(data)))
        return tuple(model.evaluate(data))
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    if ALGORITHM == "guesser":
        xTest, yTest = data
        acc = 0
        for i in range(preds.shape[0]):
            if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        accuracy = acc / preds.shape[0]
        print("Classifier algorithm: %s" % ALGORITHM)
        print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")


        return None
    else:
        raise ValueError("Algorithm not recognized.")




#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    "loss: %f\naccuracy: %f" % tuple(model.evaluate(data))
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
