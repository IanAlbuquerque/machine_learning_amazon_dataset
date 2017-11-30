""" mean difference """
# 67.68% accuracy

import numpy as np

K_CONSTANT = 50

def train(x_data, y_data, classes):
    """ trains the algorithm """

    means = []
    num_occurences = []
    for classifcation in classes:
        means.append(np.zeros((28, 28)))
        num_occurences.append(np.count_nonzero(y_data == classifcation))
    for x_value, y_value in zip(x_data, y_data):
        means[y_value] += x_value.reshape((28, 28)) / float(num_occurences[y_value])
    for mean in means:
        mean = mean.reshape(28*28)

    return means

def classify(parameters, data):
    """ tests the algorithm """
    means = parameters
    dists = [ ((data - mean.flatten()) ** 2).sum(axis=0) for mean in means ]
    return np.argmin(dists)