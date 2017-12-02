import numpy as np

"""File for testing algorithms"""

def haar_transform(data):
    """apply one step of haar trasnform"""
    data = np.int32(data)
    return np.concatenate(((data[::2] + data[1::2])/2.0, (data[::2] - data[1::2])/2.0))

def from_data_to_haar(data):
    """" from 28x28 to 32x32 haar """
    new_data = np.zeros((32, 32))
    new_data[:28, :28] = data.reshape((28, 28))

    for k in xrange(5):
        k = 5 - k
        new_data[:2**k, :2**k] = np.apply_along_axis(haar_transform, 0, new_data[:2**k, :2**k])
        new_data[:2**k, :2**k] = np.apply_along_axis(haar_transform, 1, new_data[:2**k, :2**k])

    return new_data.reshape(32*32)

def train(algorithm, x_data, y_data):
    """train algorithm"""

    percent10 = int(len(x_data)*0.1)
    # return algorithm.train(np.apply_along_axis(from_data_to_haar, 1, x_data[:percent10]), y_data[:percent10], range(10))
    return algorithm.train(x_data[:percent10], y_data[:percent10], range(10))

def test(algorithm, parameters, x_data, y_data):
    """test algorithm accuracy"""

    percent10 = int(len(x_data)*0.1)
    # return algorithm.classify(parameters,np.apply_along_axis(from_data_to_haar, 1, x_data[:percent10]),y_data)
    return algorithm.classify(parameters,x_data[:percent10],y_data[:percent10])

    # num_error = 0
    # num_correct = 0
    # for data, answer in zip(x_data, y_data):
    #     guess = algorithm.classify(parameters, data)
    #     if answer == guess:
    #         num_correct += 1
    #     else:
    #         num_error += 1

    # return float(num_correct) / float(num_error + num_correct)

def train_and_test(algorithm, x_train, y_train, x_test, y_test):
    """trains and tests algorithm accuracy"""
    parameters = train(algorithm, x_train, y_train)
    accuracy = test(algorithm, parameters, x_test, y_test)
    return accuracy

def run(path, algorithm):
    """loads, trains and tests algorithm accuracy"""
    import utils.reader

    x_train, y_train = utils.reader.load_train(path)
    x_test, y_test = utils.reader.load_test(path)

    parameters = train(algorithm, x_train, y_train)
    accuracy = test(algorithm, parameters, x_test, y_test)

    return accuracy
