""" Guesses randomly """
# 10.0% accuracy

def train(x_data, y_data, classes):
    """ trains the algorithm """
    _ = x_data
    _ = y_data
    return classes

def classify(parameters, data):
    """ tests the algorithm """
    import random

    _ = data

    classes = parameters
    return random.choice(classes)
