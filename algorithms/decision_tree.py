""" File with functions for generating a decision tree """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import functools
import numpy as np

# ============================================================================

def print_spacers(num_spaces):
    """ print n spacers """
    for _ in xrange(num_spaces):
        print('\t', end='')

class BinaryDecistionTree(object):
    """ Binary Decision Tree """
    def __init__(self):
        self.children = {}
        self.attribute = None

    def print_rec(self, level):
        """ TODO """
        for key in self.children:
            print_spacers(level)
            print('v[', self.attribute, ']', '?=', key)
            try:
                self.children[key].print_rec(level + 1)
            except AttributeError:
                print_spacers(level+1)
                print('answer =', self.children[key])

    def print(self):
        """ TODO """
        self.print_rec(0)

def importance(examples, attribute):
    """
    The importante of attribute in the given examples
    """

    _ = examples
    _ = attribute

    return 0

def plurality_value(examples):
    """
    Selects the most common output value among a set of examples, breaking ties randomly

    Keyword arguments:
        examples -- numpy array structured like
            np.zeros(N, dtype=[('data', 'Ki4'),('classification', 'i4')])
            where N and K are natural numbers
            CLASSIFICATION MUST BE NON-NEGATIVE

    Returns:
        a classification (integer)
    """

    return np.argmax(np.bincount(examples["classification"]))

def decision_tree_learning(examples, attributes, attribute_values, parent_examples):
    """
    Performs the decision tree learning algorithm

    Keyword arguments:
      examples -- numpy array structured like
        np.zeros(N, dtype=[('data', 'Ki4'),('classification', 'i4')]),
        where N and K are natural numbers
        CLASSIFICATION MUST BE NON-NEGATIVE

      attributes -- numpy array with natural numbers, those being between 0 and K-1, inclusive

      parent_examples -- TODO

    Returns:
      TODO
    """

    if examples.size == 0:
        return plurality_value(parent_examples)
    if np.all(examples["classification"] == examples["classification"][0]):
        return examples["classification"][0]
    if attributes.size == 0:
        return plurality_value(examples)

    attribute_importances = np.apply_along_axis(
        functools.partial(importance, examples), 0, attributes)
    important_attribute = attributes[np.argmax(attribute_importances)]
    tree = BinaryDecistionTree()
    tree.attribute = important_attribute
    for value in attribute_values[important_attribute]:
        new_examples = np.array([example for example in examples \
            if example["data"][important_attribute] == value])
        subtree = decision_tree_learning(new_examples, \
            np.delete(attributes, np.argwhere(attributes == important_attribute)), \
            attribute_values,                                                      \
            examples)

        tree.children[value] = subtree
    return tree

def main():
    """ The main function of the program """

    # example for testing
    arr = np.zeros(5, dtype=[('data', '3i4'), ('classification', 'i4')])
    arr['data'][0] = [1, 0, 0]
    arr['classification'][0] = 1
    arr['data'][1] = [0, 1, 0]
    arr['classification'][1] = 2
    arr['data'][2] = [0, 0, 1]
    arr['classification'][2] = 0
    arr['data'][3] = [0, 0, 2]
    arr['classification'][3] = 1
    arr['data'][4] = [1, 1, 2]
    arr['classification'][4] = 2

    decition_tree = decision_tree_learning(arr, \
        np.array(range(3)),                     \
        {0: [0, 1], 1: [0, 1], 2: [0, 1, 2]},   \
        np.array([]))

    decition_tree.print()

if __name__ == "__main__":
    main()
