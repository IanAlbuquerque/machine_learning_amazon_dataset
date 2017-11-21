""" View image by image from the dataset """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import sys
import util

# ===========================================================================

DATASET_FILE_PATH = None # "./dataset/fashion-mnist_train.csv"

# ============================================================================

def show_img(classification, pixels_matrix):
    """ Shows the image on screen """
    print(util.classification_string(classification))
    util.plot_pixels(pixels_matrix)

def main():
    """ The main function """
    util.iterate_on_pixels_matrix(DATASET_FILE_PATH, show_img)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('NOT ENOUGH ARGUMENTS!\n' + \
              'Usage:\n' + \
              '\tpython ./view_imgs.py <PATH TO INPUT FILE> ')
    else:
        DATASET_FILE_PATH = sys.argv[1]
        main()
