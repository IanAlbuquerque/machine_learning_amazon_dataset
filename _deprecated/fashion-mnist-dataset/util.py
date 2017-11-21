""" Utility functions for the dataset """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import re
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================

def classification_string(classification_idx):
    """ converts classification idx to string """
    classifications = ['T-Shirt/top', \
                       'Trouser', \
                       'Pullover', \
                       'Dress', \
                       'Coat', \
                       'Sandal', \
                       'Shirt', \
                       'Sneaker', \
                       'Bag', \
                       'Ankle boot']
    return classifications[classification_idx]

def line_to_array(line):
    """ converts line to array """
    pixels = re.split(',|\n', line)
    pixels = filter(None, pixels)
    classification = pixels[0]
    pixels = pixels[1:]
    pixels = np.array(pixels).astype(float)
    pixels = pixels/255.0
    return int(classification), pixels

def line_to_matrix(line):
    """ converts line to matrix """
    classifcation, pixels = line_to_array(line)
    pixels.shape = (28, 28)
    return classifcation, pixels

def plot_pixels(pixels):
    """ shows img on screen """
    plt.figure(1)
    plt.imshow(pixels, interpolation='nearest', cmap='gray')
    plt.grid(True)
    plt.show()

def coord_to_idx(x_coord, y_coord):
    """ converts x,y coord to corresponding linear idx """
    return x_coord * 28 + y_coord

def iterate_on_pixels_matrix(path_to_input_file, process_function):
    """ The main function """

    line_number = -1
    print('Reading JSON (' + path_to_input_file + ') file...')
    with open(path_to_input_file) as infile:
        for line in infile:
            line_number += 1
            if line_number == 0:
                continue
            classification, pixels = line_to_matrix(line)
            process_function(classification, pixels)
    print('Done!')

def iterate_on_pixels_array(path_to_input_file, process_function):
    """ The main function """

    line_number = -1
    print('Reading JSON (' + path_to_input_file + ') file...')
    with open(path_to_input_file) as infile:
        for line in infile:
            line_number += 1
            if line_number == 0:
                continue
            classification, pixels = line_to_array(line)
            process_function(classification, pixels)
    print('Done!')
