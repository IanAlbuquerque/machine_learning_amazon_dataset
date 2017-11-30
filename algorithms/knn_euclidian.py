""" KNN with Euclidian Distance """
# 75.8% accuracy

import numpy as np
from scipy import stats

K_CONSTANT = 50

def train(x_data, y_data, classes):
    """ trains the algorithm """
    return np.apply_along_axis(from_data_to_haar, 1, x_data), y_data

count = 0

def classify(parameters, data):
    """ tests the algorithm """
    global count
    x_data, y_data = parameters
    data = from_data_to_haar(data)
    distances = ((x_data-data)**2).sum(axis=1)
    # nearest = sorted(zip(distances, y_data), key=itemgetter(0))[:K_CONSTANT]
    item_indexes = np.argsort(distances)[:K_CONSTANT]
    nearest = y_data[item_indexes]
    print nearest
    return stats.mode(nearest, axis=None)[0][0]

    # def dist(xy_data):
    #     """ dist from data to x_data instance """
    #     return np.linalg.norm(data - xy_data[0])
    # _, y_k_nn = zip(*(sorted(zip(x_data, y_data), key=dist)))
    # return stats.mode(y_k_nn[:K_CONSTANT], axis=None)[0][0]

def haar_transform(data):
    """apply one step of haar trasnform"""
    data = np.int32(data)
    return np.concatenate(((data[::2] + data[1::2])/1.0, (data[::2] - data[1::2])/2.0))

def from_data_to_haar(data):
    """" from 28x28 to 32x32 haar """
    new_data = np.zeros((32, 32))
    new_data[:28, :28] = data.reshape((28, 28))

    for k in xrange(5):
        k = 5 - k
        new_data[:2**k, :2**k] = np.apply_along_axis(haar_transform, 0, new_data[:2**k, :2**k])
        new_data[:2**k, :2**k] = np.apply_along_axis(haar_transform, 1, new_data[:2**k, :2**k])

    return new_data.reshape(32*32)
