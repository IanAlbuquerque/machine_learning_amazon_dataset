""" File Description """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function
from enum import Enum

# ============================================================================

import math
import numpy as np

# ============================================================================

DATASET_FILE_PATH = "./dataset/amazon-meta.txt"

# ============================================================================

# FILE KEYWORDS
class FileKeywords:
    """ File keywords for file processing """
    id_separator = 'Id:'
    amazon_serial_index_number = 'ASIN:'
    title = 'title:'
    group = 'group:'

class Group:
    name = ''

class Category:
    csin = ''
    name = ''
    parent = None # Category

class Review:
    data = ''
    customer = ''
    rating = 0
    votes = 0
    helpful = 0

class Product:
    asin = ''
    title = ''
    group = None
    salesrank = 0
    similar = []        # Product[]
    categories = []     # Category[]
    reviews = []        # Review[]

def find_group(group_array, group_name):
    for group in group_array:
        if group_name == group.name:
            return group
    new_group = Group()
    new_group.name = group_name
    group_array.append(new_group)
    return new_group


def process_new_product_line(product_array, group_array, product_index, line_as_list):
    """ Processes a new line for a given product

    Keyword arguments:
    product_array -- the product array being build
    product_index -- the index of that product in the product array
    line_as_list -- the new line to be processes as a list
    """

    if not line_as_list:
        return

    first_word = line_as_list.pop(0)
    if first_word == FileKeywords.amazon_serial_index_number:
        product_array[product_index].asin = line_as_list[0]

    if first_word == FileKeywords.title:
        product_array[product_index].title = ' '.join(line_as_list)

    if first_word == FileKeywords.group:
        product_array[product_index].group = find_group(group_array, ' '.join(line_as_list))

def process_dataset_file(path):
    """ Processes the dataset file

    Keyword arguments:
    path -- the path to the dataset file
    """

    products = []
    groups = []
    categories = []

    is_processing_new_product = False
    product_index_being_processed = 0

    num_products_processed = 0
    num_products_expected = 548552
    percentage_done = 0
    last_printed_percentage = 0

    print('Processing dataset file ' + path + '...')
    with open(path) as infile:
        for line in infile:
            line_words = line.split()
            if not line_words:
                continue
            if line_words[0] == FileKeywords.id_separator:
                # print percentage
                num_products_processed += 1
                percentage_done = 100.0 * num_products_processed / num_products_expected
                if percentage_done - last_printed_percentage > 1.0:
                    last_printed_percentage = percentage_done
                    print(str(math.floor(percentage_done)) + '%')
                is_processing_new_product = True
                products.append(Product())
                product_index_being_processed = len(products) - 1
                continue
            if is_processing_new_product:
                process_new_product_line(products, groups, product_index_being_processed, line_words)
                continue

    print('100% complete')
    print(str(len(products)) + ' products processed')
    print(str(len(groups)) + ' groups found')
    return products, groups

def main():
    """ The main function of the program """
    products, groups = process_dataset_file(DATASET_FILE_PATH)
    for group in groups:
        print(group.name)
    """
    for product in products:
        print(product.asin, product.title)
    """

if __name__ == "__main__":
    main()
