"""Utility for reading the dataset"""

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""

    import os
    import gzip
    import numpy as np

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def load_train(path):
    """Load MNIST training data from `path`"""
    return load_mnist(path, kind='train')

def load_test(path):
    """Load MNIST training data from `path`"""
    return load_mnist(path, kind='t10k')
