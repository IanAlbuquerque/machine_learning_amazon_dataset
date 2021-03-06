""" Single layer Neural Network with MIRA """
# 76.99% accuracy

import numpy as np

# HYPER PARAMETER (MIRA PARAMETER)
STEP_MAXIMUM = 0.01

def train(x_data, y_data, classes):
    """ trains the algorithm """

    # ========================
    # initialization
    # ========================

    # work with classes indexes
    answer_indexes = [np.where(classes == answer)[0][0] for answer in y_data]

    # normalize data
    x_data = x_data / float(255.0)

    # add bias
    x_data_with_bias = np.ones((x_data.shape[0], x_data.shape[1] + 1))
    x_data_with_bias[:, :-1] = x_data

    # weights array (one for each class)
    weights = np.zeros((len(classes), x_data_with_bias.shape[1]))

    # ========================
    # the algorithm itself
    # ========================

    for data_with_bias, answer_index in zip(x_data_with_bias, answer_indexes):
        # classify examples
        guess_index = classification_index(weights, data_with_bias)
        # if right, do nothing
        # if wrong, update
        if guess_index != answer_index:
            update_scalar = correcting_update(weights[guess_index, :], \
                weights[answer_index, :], data_with_bias)
            # rotate right weight towards data
            weights[answer_index, :] += update_scalar * data_with_bias
            # rotate wrong weight away from data
            weights[guess_index, :] -= update_scalar * data_with_bias

    return classes, weights

def classify(parameters, data):
    """ tests the algorithm """
    classes, weights = parameters
    data_with_bias = np.append(data, 1.0)
    perceptron_index = classification_index(weights, data_with_bias)
    return classes[perceptron_index]

def classification_index(weights, data_with_bias):
    """ classifies the data with the given weights, returning the index of the class """
    return np.argmax([np.inner(weight, data_with_bias) for weight in weights])

def correcting_update(wrong_weight, right_weight, data):
    """ computes the increment scalar """
    suggested_scalar = (np.inner(wrong_weight - right_weight, data) + 1.0) \
        / (2.0 * np.inner(data, data))
    return min(suggested_scalar, STEP_MAXIMUM)
