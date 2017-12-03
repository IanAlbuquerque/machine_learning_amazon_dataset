""" View image by image from the dataset """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

import sys
from os import listdir

# ============================================================================

import utils.reader
import utils.viewer
import utils.algorithms
# import pkgutil

# ============================================================================

# LOADS ALL ALGORITHMS
from algorithms import __all__ as algorithms_submodules
ALGORITHMS = __import__('algorithms', globals()['algorithms_submodules'], locals(), algorithms_submodules, 0)

# ============================================================================

def print_usage_message():
    """ Prints file usage message """
    print('Usage:\n' + \
            '\tpython ./run.py <algorithm_name>\n' + \
            'Example:\n' + \
            '\tpython ./run.py dumb\n' + \
            'Available Algorithms:')
    for file_name in listdir('algorithms'):
        if file_name.endswith('.py') and not file_name.startswith('_'):
            print('\t' + file_name[:-3])

def main():
    """ The main function """
    if len(sys.argv) != 2:
        print_usage_message()
    elif not sys.argv[1] + '.py' in listdir('algorithms'):
        print_usage_message()
    else:
        accuracy = utils.algorithms.run('data', getattr(ALGORITHMS, sys.argv[1]))
        print('Accuracy = ' + str(accuracy * 100) + ' %')

# ============================================================================

if __name__ == "__main__":
    main()
