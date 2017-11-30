"""File for testing algorithms"""

def train(algorithm, x_data, y_data):
    """train algorithm"""
    return algorithm.train(x_data, y_data, range(10))

def test(algorithm, parameters, x_data, y_data):
    """test algorithm accuracy"""

    num_error = 0
    num_correct = 0
    for data, answer in zip(x_data, y_data):
        guess = algorithm.classify(parameters, data)
        if answer == guess:
            num_correct += 1
        else:
            num_error += 1
        print(100.0 * float(num_correct) / max(1.0,float(num_error + num_correct)))

    return float(num_correct) / float(num_error + num_correct)

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
