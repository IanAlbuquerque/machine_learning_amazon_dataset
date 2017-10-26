""" File with functions for generating a decision tree """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import functools
import math
import numpy as np

# ============================================================================

def print_spacers(num_spaces):
    """ print n spacers """
    for _ in xrange(num_spaces):
        print('\t', end='')

class NaryDecisionTree(object):
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

def entropy(examples):
    """ TODO """

    value_class_size = [
        (examples['classification'] == k).sum()
        for k in np.unique(examples['classification'])
        ]

    result = 0.0
    total_size = examples.size

    for size in value_class_size:
        probability = float(size) / float(total_size)
        result += probability * math.log(probability, 2.0)

    return - result

def importance(examples, attribute):
    """
    The importante of attribute in the given examples
    """

    attribute_classes = [
        examples[examples['data'][:, attribute] == k]
        for k in np.unique(examples['data'][:, attribute])
        ]

    remainder = 0.0
    total_size = examples.size

    for attribute_class in attribute_classes:
        probability = float(attribute_class.size) / float(total_size)
        remainder += probability * entropy(attribute_class)

    gain = entropy(examples) - remainder

    return gain

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

    vectorized_importance = np.vectorize(functools.partial(importance, examples))
    attribute_importances = vectorized_importance(attributes)
    important_attribute = attributes[np.argmax(attribute_importances)]
    tree = NaryDecisionTree()
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

    
    restaurant = np.zeros(12, dtype=[('data', '10i4'), ('classification', 'i4')])
    restaurant['data'][0]   = [1, 0, 0, 1, 1, 2, 0, 1, 0, 0]
    restaurant['data'][1]   = [1, 0, 0, 1, 2, 0, 0, 0, 1, 2]
    restaurant['data'][2]   = [0, 1, 0, 0, 1, 0, 0, 0, 2, 0]
    restaurant['data'][3]   = [1, 0, 1, 1, 2, 0, 1, 0, 1, 1]
    restaurant['data'][4]   = [1, 0, 1, 0, 2, 2, 0, 1, 0, 3]
    restaurant['data'][5]   = [0, 1, 0, 1, 1, 1, 1, 1, 3, 0]
    restaurant['data'][6]   = [0, 1, 0, 0, 0, 0, 1, 0, 2, 0]
    restaurant['data'][7]   = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    restaurant['data'][8]   = [0, 1, 1, 0, 2, 0, 1, 0, 2, 3]
    restaurant['data'][9]   = [1, 1, 1, 1, 2, 2, 0, 1, 3, 1]
    restaurant['data'][10]  = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    restaurant['data'][11]  = [1, 1, 1, 1, 2, 0, 0, 0, 2, 2]
    restaurant['classification'][0]     = 1
    restaurant['classification'][1]     = 0
    restaurant['classification'][2]     = 1
    restaurant['classification'][3]     = 1
    restaurant['classification'][4]     = 0
    restaurant['classification'][5]     = 1
    restaurant['classification'][6]     = 0
    restaurant['classification'][7]     = 1
    restaurant['classification'][8]     = 0
    restaurant['classification'][9]     = 0
    restaurant['classification'][10]    = 0
    restaurant['classification'][11]    = 1

    attribute_values = {}
    attribute_values[0] = range(2) # Alt
    attribute_values[1] = range(2) # Bar
    attribute_values[2] = range(2) # Fri
    attribute_values[3] = range(2) # Hun
    attribute_values[4] = range(3) # Pat [none, some, full]
    attribute_values[5] = range(3) # Price [$, $$, $$$]
    attribute_values[6] = range(2) # Rain
    attribute_values[7] = range(2) # Res
    attribute_values[8] = range(4) # Type [French, Thai, Burger, Italian]
    attribute_values[9] = range(4) # Est [0-10, 10-30, 30-60, >60]

    decision_tree = decision_tree_learning(restaurant, \
        np.array(range(10)), attribute_values, np.array([]))

    decision_tree.print()

    decision_tree = decision_tree_learning(arr, \
        np.array(range(3)),                     \
        {0: [0, 1], 1: [0, 1], 2: [0, 1, 2]},   \
        np.array([]))

    decision_tree.print()

if __name__ == "__main__":
    main()
