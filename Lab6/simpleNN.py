import keras.layers as Layers
import keras.optimizers as Optimizers
from keras import Sequential
import numpy as np
import random

def createNewSeqModel():
    model = Sequential()
    weights = 'random_uniform'

    model.add(Layers.Dense(50, input_shape=(1,), activation='relu', kernel_initializer=weights))

    model.add(Layers.Dense(50, activation='relu', kernel_initializer=weights))

    model.add(Layers.Dense(1,activation='linear', kernel_initializer=weights))

    model.compile(Optimizers.Adam(), loss='mse')
    return model

def createTrainingData(amount, maxNum, func):
    inData = [random.randint(0, maxNum) for i in range(amount)]
    results = [func(d) for d in inData]
    return inData, results

def predictWithModel(model, samples):
    return model.predict(np.array(samples))

def trainModel(model, samples, labels, epochs, batchSize):
    model.fit(np.array(samples), np.array(labels), epochs = epochs, batch_size = batchSize)

func = lambda x: x*3
samples, labels = createTrainingData(10000, 1000, func)
evalSamples, evalLabels = createTrainingData(5, 100, func)
print(evalSamples)
print(evalLabels)

myNet = createNewSeqModel()

trainModel(myNet, samples, labels, 100, 100)

evalPrediction = predictWithModel(myNet, evalSamples)
print("Prediction:\n", evalPrediction)
print("Labels:\n", evalLabels)
